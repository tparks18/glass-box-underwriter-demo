# Glass Box Underwriter Streamlit Demo

This is a lightweight What-If demo for the Glass Box Underwriter project.

## What it does

The app lets a user change applicant inputs, such as age, BMI, smoker status, sex, children, and region. It then shows whether the model predicts the applicant as Good Risk or Bad Risk.

The target is:

```text
Bad Risk = annual charges > $10,000
Good Risk = annual charges <= $10,000
```

## Local setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

On Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload these files.
3. Run `python train_model.py` locally first so the `models/` folder exists.
4. Push the repo to GitHub.
5. Go to Streamlit Community Cloud.
6. Create a new app.
7. Select the repository.
8. Set the main file path to `app.py`.
9. Deploy.
10. Put the deployed app link or QR code in your presentation.

## Presentation language

Because the original Google What-If Tool did not run reliably in the Kaggle environment, we built a smaller version of the same idea. The demo lets users change applicant inputs and immediately see how the model's risk prediction changes. This makes the underwriting model more transparent because the audience can test the model's behavior instead of only seeing a static prediction.