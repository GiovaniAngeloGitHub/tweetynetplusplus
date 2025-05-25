import os
import pandas as pd
import json
from sklearn.metrics import classification_report


def save_classification_report(y_true, y_pred, class_names, output_dir="logs", as_csv=True, as_json=True):
    os.makedirs(output_dir, exist_ok=True)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    base_filename = "classification_report"

    if as_csv:
        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
        pd.DataFrame(report_dict).transpose().to_csv(csv_path)
        print(f"📄 classification_report salvo em {csv_path}")
    if as_json:
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, "w") as f:
            json.dump({str(k): v for k, v in report_dict.items()}, f, indent=4)
        print(f"📝 classification_report salvo em {json_path}")
