# Fuzzy Logic Inference System App

A modern, interactive Streamlit application for building, visualising, and experimenting with both Type-1 and Interval Type-2 (IT2) fuzzy logic systems. Ideal for education, research, and rapid prototyping of fuzzy inference systems.

---

## Features

- **Graphical Fuzzy System Builder:** Define input/output variables, create fuzzy sets, and build rules visually.
- **Supports Type-1 and IT2 Fuzzy Logic:** Switch between classic and interval type-2 fuzzy systems.
- **Live Visualisation:** See membership functions, rule aggregation, and inference results instantly.
- **Export & Save:** Download your fuzzy system as a runnable Python script or as a JSON config.
- **Interactive CLI Exports:** Exported Python scripts prompt for input and show results in the terminal.
- **Clean, Modern UI:** Built with Streamlit for ease of use and rapid iteration.

---

## Quick Start

### 1. Install Requirements

It's recommended to use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

---

## Usage

- Use the sidebar to select **Type 1 Fuzzy** or **Type 2 Fuzzy**.
- Define your variables, fuzzy sets, and rules.
- Visualise membership functions and inference results.
- Save your fuzzy system as JSON or export as a Python script.
- Load saved configs to continue working later.

---

## Project Structure

```
.
├── app.py                # Streamlit app entrypoint
├── requirements.txt
├── pages/                # Streamlit multipage entrypoints
│   ├── type1_fuzzy.py
│   └── type2_fuzzy.py
├── src/
│   ├── core/
│   │   ├── fuzzy_utils.py
│   │   ├── type1/
│   │   │   ├── fuzzy_fuzzification.py
│   │   │   └── fuzzy_inference.py
│   │   └── type2/
│   │       ├── fuzzy_fuzzification_it2.py
│   │       └── fuzzy_inference_it2.py
│   ├── ui/
│   │   ├── type1_fuzzy_ui_sections.py
│   │   └── type2_fuzzy_ui_sections.py
│   └── export/
│       ├── fuzzy_export.py
│       └── fuzzy_export_it2.py
├── tests/                # Test scripts
│   ├── test_type1_interactive.py
│   ├── test_it2_fixed.py
│   └── test_it2_interactive.py
└── .streamlit/           # Streamlit config (optional)
```

---

## Contributing

Pull requests and suggestions are welcome! Please open an issue or PR.

---

## Licence

MIT Licence

---


