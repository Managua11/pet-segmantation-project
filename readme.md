შინაური ცხოველების სეგმენტაცია Oxford-IIIT Pet მონაცემებზებინარული სეგმენტაციის მოდელი (ცხოველი vs ფონი) Oxford-IIIT Pet მონაცემთა ბაზისთვის, შექმნილი U-Net არქიტექტურითა და ImageNet-ზე წინასწარ გაწვრთნილი ResNet18 ენკოდერით, სუფთა PyTorch-ზე CPU-ზე სამუშაოდ.ცოცხალი დემოდემო ბმული: https://managua11.github.io/pet-segmentation/14.4 მბ ზომის INT8 ONNX მოდელი ეშვება პირდაპირ ბრაუზერში ONNX Runtime Web (WebAssembly) ტექნოლოგიით. სერვერი არ არის საჭირო და ფოტო არ ტოვებს მომხმარებლის მოწყობილობას. დემოში შესაძლებელია საკუთარი ფოტოს ატვირთვა ან ტესტური ნაკრების 3 ნიმუშის ცდა, ზღურბლისა (threshold) და გამჭვირვალობის (opacity) სლაიდერებით, ასევე Original / Overlay / Cutout ხედებით. გვერდის კოდი განთავსებულია docs/ საქაღალდეში და გაშვებულია GitHub Pages-ით.შედეგებიმოდელიVal IoUVal Diceვარჯიშის დრო (CPU)სრული ფაინ-თუნინგი (baseline)0.87020.927674 წთგაყინული ენკოდერი (საბოლოო მოდელი)0.89660.943652 წთშეფასება ოფიციალურ ტესტურ ნაკრებზე (3662 სურათი, შემოწმებული ერთხელ): IoU 0.9003, Dice 0.9456.ოპტიმიზაციის ბენჩმარკი (CPU, batch size 1):PyTorch FP32: 43.2 მწმ / 57.4 მბONNX FP32: 20.4 მწმ (2.1-ჯერ სწრაფი) / 57.3 მბONNX INT8: 37.4 მწმ / 14.4 მბ (4-ჯერ პატარა), Val IoU 0.8960სწრაფი დაწყებაინსტალაცია (Windows / Python 3.14, CPU)python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
შენიშვნები:Linux/macOS-ზე ვირტუალური გარემოს აქტივაცია ხდება ბრძანებით: source .venv/bin/activate.თუ კორპორატიული ქსელის SSL შეცდომის გამო ImageNet-ის წონების ჩამოტვირთვა ვერ ხერხდება, ვარჯიშის დაწყებამდე გაშვით import truststore; truststore.inject_into_ssl(), ან გამოიყენეთ სხვა ქსელი.Windows-ზე import onnxruntime-ის ხარვეზით გათიშვისას დააინსტალირეთ Microsoft Visual C++ Redistributable. torch ბიბლიოთეკის იმპორტი უნდა მოხდეს onnxruntime-ზე ადრე (კოდში ეს ავტომატურად გათვალისწინებულია).მონაცემების მომზადებაჩამოტვირთეთ images.tar.gz და annotations.tar.gz ბმულიდან https://www.robots.ox.ac.uk/~vgg/data/pets/ და ამოაარქივეთ შემდეგ სტრუქტურაში:data/oxford-iiit-pet/
├── images/            # *.jpg
└── annotations/
    ├── trimaps/       # *.png
    ├── trainval.txt
    └── test.txt
კოდი ავტომატურად ფილტრავს 7 ცარიელ ნიღაბს trainval-ში და 7-ს ტესტში. მონაცემები გაყოფილია 85/15 პროპორციით (seed 42), რაც იძლევა 3122 საწვრთნელ და 551 ვალიდაციურ სურათს.წონების მომზადებაჩამოტვირთეთ best.pt GitHub Release-დან ბმულით:
WEIGHTS_URL
და განათავსეთ მისამართზე: artifacts/frozen_encoder/best.pt.გამოყენებაყველა ბრძანება გააშვით პროექტის ძირითადი საქაღალდიდან (repo root):# მოდელის გაწვრთნა (წერს ლოგებს experiments/<name>/log.csv-ში და წონებს artifacts/<name>/best.pt-ში)
python -m src.train --exp-name baseline
python -m src.train --exp-name frozen_encoder --freeze-encoder

# შეფასება (წერს per_image.csv, summary.json და worst_10.png ფაილებს experiments/<name>/eval_<split>/-ში)
python -m src.evaluate --checkpoint artifacts/frozen_encoder/best.pt --split val
python -m src.evaluate --checkpoint artifacts/frozen_encoder/best.pt --split test

# ოპტიმიზაცია: ONNX FP32 + INT8 ექსპორტი და ბენჩმარკი (წერს experiments/optimization/results.md-ს)
python -m src.optimize --checkpoint artifacts/frozen_encoder/best.pt

# ლოკალური Streamlit დემო აპლიკაცია (საჭიროებს artifacts/frozen_encoder/best.pt ფაილს)
streamlit run app.py

# ტესტების გაშვება
python -m pytest tests
დამატებითი პარამეტრები: --epochs (ნაგულისხმევი 10), --lr (ნაგულისხმევი 1e-3), --batch-size (ნაგულისხმევი 16). ვარჯიშის ერთი ციკლი 6-ბირთვიან CPU-ზე დაახლოებით 50–75 წუთს სჭირდება.პროექტის სტრუქტურაsrc/
  dataset.py    # PetDataset, ცარიელი ნიღბების გაფილტვრა, train/val გაყოფა
  model.py      # U-Net + ResNet18 (segmentation_models_pytorch)
  metrics.py    # IoU, Dice
  train.py      # საწვრთნელი ციკლი
  evaluate.py   # მეტრიკები + შეცდომების ანალიზი (სახეობა, ზომა, ჯიშები, ცუდი სურათები)
  predict.py    # ინფერენსი ერთი სურათისთვის
  optimize.py   # ONNX ექსპორტი, INT8 კვანტიზაცია, ბენჩმარკი
app.py          # Streamlit დემო
docs/           # ბრაუზერის დემო (GitHub Pages)
notebooks/01_data_exploration.ipynb   # მონაცემების ანალიზი
experiments/    # ლოგები, კონფიგურაციები, შეფასების შედეგები, ოპტიმიზაციის შედეგები
tests/test_core.py
დოკუმენტაცია: REPORT.md