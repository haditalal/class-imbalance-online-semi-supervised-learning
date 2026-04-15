import os
import csv
import glob

# 📁 مجلد الملفات
results_dir = "."  # عدّله لو ملفاتك بمجلد آخر

# 🔸 أسماء الأعمدة حسب ترتيب القيم في السطر
header = [
    "Dataset",
    "LabelingStrategy",
    "LabelingRatio",
    "ImbalanceRatio",
    "H",
    "N",
    "lam",
    "alpha",
    "beta",
    "gamma",
    "Best_GMean"
]

# 📝 اسم الملف النهائي
output_file = "Tuning_All_Parameters.csv"

# 🧭 الحصول على كل الملفات التي تبدأ بـ tuning_result_case_
files = sorted(glob.glob(os.path.join(results_dir, "tuning_result_case_*.txt")))

# 🧾 دمج المحتوى
with open(output_file, "w", newline="") as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(header)  # كتابة العنوان مرة واحدة
    
    for file_path in files:
        with open(file_path, "r") as f:
            line = f.readline().strip()
            if line:
                values = line.split(",")
                writer.writerow(values)

print(f"✅ تم تجميع {len(files)} ملف في: {output_file}")
