import io
import math
import re

import pandas as pd
import streamlit as st
from pypdf import PdfReader


st.set_page_config(
    page_title="Horse Racing PDF Assistant",
    layout="wide",
)

st.title("Horse Racing PDF Assistant")
st.caption("Upload a racecard PDF, check the runners, and analyse the market.")

st.info(
    "PDF only — no historical CSV required. This version does not invent "
    "historical form or confidence scores. Probabilities are derived from "
    "the odds in the racecard, not from a trained prediction model."
)


# -------------------------------------------------
# PDF extraction and parsing
# -------------------------------------------------

RACE_HEADER = re.compile(
    r"\bRACE\s+(\d{1,2})\b",
    re.IGNORECASE,
)

DISTANCE = re.compile(
    r"\b(\d{3,4})\s*m\b",
    re.IGNORECASE,
)

# Supported examples:
# 1 (3) THUNDER STRIKE 60.0 R Fourie  J Snaith  4/1
# 1 THUNDER STRIKE (3) 60.0 R Fourie  J Snaith  5.00
RUNNER = re.compile(
    r"^\s*(?P<number>\d{1,2})\s+"
    r"(?:\((?P<draw_before>\d{1,2})\)\s+)?"
    r"(?P<horse>[^\d()]+?)\s+"
    r"(?:\((?P<draw_after>\d{1,2})\)\s+)?"
    r"(?P<weight>\d{2}(?:\.\d{1,2})?)"
    r"(?:\s+(?P<details>.*))?$"
)

TRAILING_ODDS = re.compile(
    r"(?:^|\s)"
    r"(?P<odds>\d+\s*/\s*\d+|\d+(?:\.\d+)?)"
    r"\s*(?:JF|CF|F)?\s*$",
    re.IGNORECASE,
)

COLUMNS = [
    "Race",
    "Number",
    "Horse",
    "Draw",
    "Weight kg",
    "Jockey",
    "Trainer",
    "Decimal odds",
    "Active",
]


def parse_odds(value):
    """Fractional odds convert to decimal. Plain numbers mean decimal."""
    if value is None:
        return None

    text = re.sub(r"\s+", "", str(value)).upper()
    text = re.sub(r"(JF|CF|F)$", "", text)

    if text in {"EVS", "EVENS", "EVEN"}:
        return 2.0

    try:
        if "/" in text:
            numerator, denominator = text.split("/")
            numerator = float(numerator)
            denominator = float(denominator)

            if numerator <= 0 or denominator <= 0:
                return None

            result = 1.0 + numerator / denominator
        else:
            result = float(text)

        if not math.isfinite(result) or result <= 1:
            return None

        return result

    except (ValueError, TypeError, ZeroDivisionError):
        return None


@st.cache_data(show_spinner=False)
def extract_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))

    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise ValueError("This PDF is password-protected.")

    if len(reader.pages) > 150:
        raise ValueError("Please use a PDF with 150 pages or fewer.")

    pages = []

    for page in reader.pages:
        try:
            text = page.extract_text(extraction_mode="layout")
        except TypeError:
            text = page.extract_text()

        pages.append(text or "")

    return "\n".join(pages)


def parse_racecard(text):
    rows = []
    seen = set()
    current_race = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        header = RACE_HEADER.search(line)

        if header:
            current_race = f"Race {int(header.group(1))}"
            distance = DISTANCE.search(line)

            if distance:
                current_race += f" - {distance.group(1)}m"

            continue

        if current_race is None:
            continue

        match = RUNNER.match(line)

        if not match:
            continue

        horse = re.sub(r"\s+", " ", match.group("horse")).strip()

        if len(horse) < 2:
            continue

        number = int(match.group("number"))
        key = (current_race.split(" - ")[0], number)

        if key in seen:
            continue

        seen.add(key)

        draw_text = (
            match.group("draw_before")
            or match.group("draw_after")
        )

        details = match.group("details") or ""
        odds_match = TRAILING_ODDS.search(details)
        odds = None

        if odds_match:
            odds = parse_odds(odds_match.group("odds"))
            details = details[:odds_match.start()].strip()

        # Only split names when actual column spacing is present.
        # Do not guess how multiword surnames should be divided.
        name_columns = [
            part.strip()
            for part in re.split(r"\s{2,}|\t+", details)
            if part.strip()
        ]

        jockey = name_columns[0] if len(name_columns) >= 2 else ""
        trainer = name_columns[1] if len(name_columns) >= 2 else ""

        rows.append(
            {
                "Race": current_race,
                "Number": number,
                "Horse": horse,
                "Draw": int(draw_text) if draw_text else None,
                "Weight kg": float(match.group("weight")),
                "Jockey": jockey,
                "Trainer": trainer,
                "Decimal odds": odds,
                "Active": True,
            }
        )

    return pd.DataFrame(rows, columns=COLUMNS)


# -------------------------------------------------
# Analysis
# -------------------------------------------------

def show_analysis(data):
    data = data.copy()

    data["Race"] = data["Race"].fillna("").astype(str).str.strip()
    data["Horse"] = data["Horse"].fillna("").astype(str).str.strip()
    data["Decimal odds"] = pd.to_numeric(
        data["Decimal odds"], errors="coerce"
    )

    data = data[
        data["Active"].fillna(False).astype(bool)
        & data["Race"].ne("")
        & data["Horse"].ne("")
    ]

    if data.empty:
        st.warning("Add at least one active runner with a race and horse name.")
        return

    for race, runners in data.groupby("Race", sort=False):
        st.subheader(race)

        runners = runners.copy()

        if runners["Number"].duplicated().any():
            st.error(
                "Duplicate runner numbers found. Correct them before analysing."
            )
            continue

        odds = runners["Decimal odds"]
        valid_odds = odds.map(
            lambda value: (
                pd.notna(value)
                and math.isfinite(float(value))
                and float(value) > 1
            )
        )

        st.caption(f"{len(runners)} active runners in the edited table.")

        if not valid_odds.all():
            st.warning(
                "One or more runners have missing or invalid decimal odds. "
                "No race probabilities are calculated until all active runners "
                "have odds greater than 1."
            )
            st.dataframe(runners, hide_index=True, use_container_width=True)
            continue

        inverse_odds = 1.0 / odds
        book_total = inverse_odds.sum()

        runners["Raw implied %"] = inverse_odds * 100
        runners["Normalised market %"] = inverse_odds / book_total * 100
        runners["Market reference odds"] = book_total * odds

        runners = runners.sort_values(
            "Normalised market %", ascending=False
        ).reset_index(drop=True)

        runners.insert(0, "Market rank", range(1, len(runners) + 1))

        st.metric("Implied book total", f"{book_total * 100:.2f}%")

        st.caption(
            "Normalised market % rescales the supplied odds to total 100%. "
            "Market reference odds are derived from that same market; "
            "they are not independently estimated fair odds or value signals."
        )

        st.dataframe(
            runners.round(2),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download this race",
            data=runners.to_csv(index=False).encode("utf-8"),
            file_name=f"{race.replace(' ', '_')}_analysis.csv",
            mime="text/csv",
            key=f"download_{race}",
        )


# -------------------------------------------------
# Interface
# -------------------------------------------------

uploaded = st.file_uploader("Upload a racecard PDF", type=["pdf"])

if uploaded is not None:
    file_bytes = uploaded.getvalue()

    if len(file_bytes) > 20 * 1024 * 1024:
        st.error("Please upload a PDF smaller than 20 MB.")
        st.stop()

    try:
        with st.spinner("Extracting racecard text..."):
            extracted_text = extract_pdf(file_bytes)

    except Exception as exc:
        st.error(f"Could not read the PDF: {exc}")
        st.stop()

    if not extracted_text.strip():
        st.error(
            "No readable text was found. This may be a scanned PDF. "
            "This version does not include OCR."
        )
        st.stop()

    with st.expander("View extracted PDF text"):
        st.text_area(
            "Extracted text",
            value=extracted_text,
            height=300,
            disabled=True,
        )

    parsed = parse_racecard(extracted_text)

    if parsed.empty:
        st.warning(
            "The layout was not recognised. You can add runners manually "
            "in the table below. Use the extracted text for reference."
        )

    st.subheader("Check and correct runners")

    st.warning(
        "PDF extraction can omit or misread runners. Check every race against "
        "the PDF, add missing horses, correct odds, and untick scratched runners. "
        "A partial field will give misleading percentages."
    )

    st.caption(
        "Use decimal odds: 4/1 = 5.00; 5/2 = 3.50. "
        "Jockey and trainer names are left blank if column boundaries "
        "cannot be identified reliably."
    )

    edited = st.data_editor(
        parsed,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Race": st.column_config.TextColumn(required=True),
            "Horse": st.column_config.TextColumn(required=True),
            "Number": st.column_config.NumberColumn(
                min_value=1, step=1
            ),
            "Draw": st.column_config.NumberColumn(
                min_value=1, step=1
            ),
            "Weight kg": st.column_config.NumberColumn(
                min_value=0.0, step=0.5
            ),
            "Decimal odds": st.column_config.NumberColumn(
                min_value=1.01, step=0.01, format="%.2f"
            ),
            "Active": st.column_config.CheckboxColumn(default=True),
        },
        key=f"runners_{uploaded.name}_{len(file_bytes)}",
    )

    confirmed = st.checkbox(
        "I have checked the complete field, odds, and scratchings against the PDF."
    )

    if st.button("Analyse checked runners", type="primary"):
        if not confirmed:
            st.warning("Check the runners and tick the confirmation first.")
        else:
            show_analysis(edited)

else:
    st.write("Upload your racecard to begin.")

st.divider()
st.caption(
    "Research tool only. No automatic bets, historical data feed, "
    "trained prediction model, or independently calculated betting edge."
)
