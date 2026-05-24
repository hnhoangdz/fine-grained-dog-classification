# Fine-Grained Dog Classification — Tổng hợp chi tiết theo tuần

## Tổng quan dự án

Dự án phân loại chi tiết giống chó (fine-grained image classification) trên bộ dữ liệu **Stanford Dogs** gồm **20.580 ảnh** thuộc **120 giống chó**. Pipeline hoàn chỉnh từ khám phá dữ liệu, tiền xử lý, huấn luyện baseline, transfer learning, đánh giá toàn diện đến kiểm thử độ ổn định hệ thống.

**Công nghệ sử dụng:** Python 3.12+, PyTorch, torchvision, scikit-learn, matplotlib, PIL

**Môi trường:** NVIDIA GPU (khuyến nghị), `uv` để quản lý dependencies

---

## Week 4 — Khám phá dữ liệu

### Tasks đã thực hiện

1. **Tải dữ liệu từ Kaggle** — Download bộ Stanford Dogs Dataset qua `kagglehub`
2. **Thống kê tổng quan** — Đếm tổng samples, số classes, số ảnh mỗi class
3. **Phân tích phân phối** — Vẽ biểu đồ bar chart và histogram phân phối ảnh theo class
4. **Đánh giá mức độ cân bằng** — Tính tỷ lệ mất cân bằng max/min
5. **Visualize mẫu dữ liệu** — Hiển thị ảnh mẫu từ 25 giống chó ngẫu nhiên

### Output nhận được

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số samples | 20.580 |
| Tổng số classes (giống chó) | 120 |
| Trung bình ảnh/class | ~171.5 |
| Class nhiều ảnh nhất | Maltese_dog (252 ảnh) |
| Class ít ảnh nhất | redbone (148 ảnh) |
| Tỷ lệ mất cân bằng (max/min) | ~1.70x |

**Kết luận:** Dữ liệu tương đối cân bằng giữa các class, không cần áp dụng oversampling/undersampling. Phần lớn class nằm trong khoảng 148–200 ảnh. Thách thức chính nằm ở bản chất fine-grained — các giống chó có sự khác biệt rất nhỏ về ngoại hình.

---

## Week 5 — Tiền xử lý dữ liệu và xây dựng pipeline

### Tasks đã thực hiện

1. **Chuẩn hóa đường dẫn** — Thiết lập DATASET_ROOT, IMAGES_ROOT, ANNOTATIONS_ROOT
2. **Tạo manifest dữ liệu** — Mỗi sample được quy về record gồm: `image_path`, `annotation_path`, `class_folder`, `breed_name`, `class_id`
3. **Parse XML annotation** — Trích xuất bounding box (xmin, ymin, xmax, ymax) và kích thước ảnh từ file annotation XML
4. **Chia train/val/test theo stratified split** — Chia theo từng class để đảm bảo phân phối đồng đều
5. **Định nghĩa augmentation pipeline**
   - `train_transform`: RandomResizedCrop(224) → RandomHorizontalFlip → ColorJitter → RandomRotation(10) → Normalize(ImageNet)
   - `eval_transform`: Resize(256) → CenterCrop(224) → Normalize(ImageNet)
6. **Xây dựng Custom PyTorch Dataset** — Class `StanfordDogsDataset` hỗ trợ bật/tắt bbox crop
7. **Tạo DataLoader** — Cho cả 3 split train/val/test
8. **Sanity check** — Kiểm tra shape, dtype, giá trị min/max sau normalize
9. **So sánh ảnh gốc vs ảnh crop theo bbox** — Hỗ trợ quyết định USE_BBOX
10. **Visualize augmentation** — Hiển thị ảnh gốc + bbox, base input, và nhiều phiên bản augmented

### Output nhận được

| Split | Samples | Classes | Min/class | Max/class |
|-------|---------|---------|-----------|-----------|
| train | 14.358 | 120 | 103 | 176 |
| val | 3.078 | 120 | 22 | 37 |
| test | 3.144 | 120 | 22 | 39 |

**Tỷ lệ chia:** 70% train / 15% val / 15% test

**Artifacts được tạo:**
- `artifacts/datasets/class_to_idx.json`
- `artifacts/datasets/train_records.json`
- `artifacts/datasets/val_records.json`
- `artifacts/datasets/test_records.json`

**Batch shape:** `(32, 3, 224, 224)` — tensor dtype float32, giá trị sau normalize nằm trong khoảng hợp lệ.

---

## Week 6-7 — AlexNet from Scratch

### Tasks đã thực hiện

1. **Nghiên cứu kiến trúc AlexNet** — Đối chiếu từ paper gốc (Krizhevsky et al., 2012) và source code torchvision
2. **Implement AlexNet bằng `torch.nn.Module`** — Tự viết toàn bộ kiến trúc, không dùng `torchvision.models.alexnet()`
   - Features: 5 Conv layers (64→192→384→256→256), MaxPool, ReLU
   - Classifier: AdaptiveAvgPool2d(6,6) → Dropout(0.5) → FC(9216→4096) → FC(4096→4096) → FC(4096→120)
   - Weight initialization theo paper gốc (bias = 1 cho conv2, conv4, conv5, fc1, fc2)
3. **Hỗ trợ 2 chế độ normalization** — `imagenet` (mean/std ImageNet chuẩn) hoặc `dataset-specific` (tự tính và cache)
4. **Thiết lập training loop** — SGD + CrossEntropyLoss + ReduceLROnPlateau + AMP (mixed precision)
5. **Huấn luyện 20 epochs** — Lưu checkpoint tốt nhất theo val_top1
6. **Vẽ training curves** — Loss curves và Top-1/Top-5 accuracy curves
7. **Đánh giá trên test set** — Load best checkpoint và đo metrics
8. **Visualize predictions** — Hiển thị ảnh test với nhãn thực vs nhãn dự đoán

### Cấu hình huấn luyện

| Hyperparameter | Giá trị |
|----------------|---------|
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Dropout | 0.5 |
| Batch size | 32 |
| Epochs | 20 |
| Scheduler | ReduceLROnPlateau (patience=2, factor=0.1) |
| AMP | Enabled (on GPU) |
| USE_BBOX | False |
| Augmentation | Resize(256) → RandomCrop(224) → RandomHorizontalFlip |

### Output nhận được

| Chỉ số | Giá trị |
|--------|---------|
| Trainable params | ~57.0M |
| Output shape | (batch, 120) |

**Kết quả test set** (best checkpoint):

AlexNet from scratch đạt kết quả baseline — accuracy thấp do train từ random initialization trên dataset nhỏ (chỉ ~170 ảnh/class) với kiến trúc cũ. Đây là baseline để so sánh với transfer learning.

**Artifacts được tạo:**
- `artifacts/checkpoints/alexnet_from_scratch_best.pt`
- `artifacts/training/alexnet_from_scratch_history.json`

---

## Week 8-9 — Transfer Learning: ResNet50, MobileNetV2, EfficientNet-B0

### Tasks đã thực hiện

1. **Thiết kế model factory** — Hàm `build_transfer_model()` hỗ trợ 3 backbone:
   - **ResNet50** — 25M params, thay `model.fc`
   - **MobileNetV2** — lightweight, thay `model.classifier`
   - **EfficientNet-B0** — compound scaling, thay `model.classifier`
2. **Chiến lược huấn luyện 2 pha:**
   - **Phase 1 — Feature Extraction (5 epochs):** Freeze toàn bộ backbone, chỉ train classifier head mới. BatchNorm giữ ở eval mode để bảo toàn pretrained statistics. AdamW, LR = 1e-3.
   - **Phase 2 — Fine-tuning (15 epochs):** Unfreeze backbone, train end-to-end với differential learning rate. Backbone LR = 1e-5, Head LR = 1e-4. CosineAnnealingLR scheduler.
3. **Label smoothing 0.1** — Giảm overconfidence trên dataset nhỏ
4. **Augmentation mạnh hơn** — Thêm RandomResizedCrop(scale=0.8-1.0) và ColorJitter so với AlexNet
5. **Bật USE_BBOX = True** — Crop theo bounding box annotation với padding 5%
6. **Vẽ training curves** — So sánh loss/accuracy cả 3 backbone trên cùng figure, đường kẻ dọc đánh dấu ranh giới Phase 1 → Phase 2
7. **Đánh giá trên test set** — Load best checkpoint mỗi backbone, đo Top-1/Top-5
8. **Bảng so sánh tổng hợp** — So sánh cả 3 backbone + AlexNet from scratch
9. **Visualize predictions** — Chọn model có val_top1 cao nhất, hiển thị dự đoán trên test set

### Cấu hình huấn luyện

| Hyperparameter | Phase 1 | Phase 2 |
|----------------|---------|---------|
| Optimizer | AdamW | AdamW |
| LR (backbone) | — (frozen) | 1e-5 |
| LR (head) | 1e-3 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Label smoothing | 0.1 | 0.1 |
| Epochs | 5 | 15 |
| Batch size | 32 | 32 |
| USE_BBOX | True | True |
| BBOX_PADDING | 0.05 | 0.05 |

### Output nhận được

**Thông số model:**

| Backbone | Total params | Trainable (Phase 1) |
|----------|-------------|---------------------|
| ResNet50 | ~25M | ~245K (head only) |
| MobileNetV2 | ~3.5M | ~154K (head only) |
| EfficientNet-B0 | ~5.3M | ~154K (head only) |

Transfer learning cải thiện rõ rệt so với AlexNet from scratch nhờ tận dụng pretrained ImageNet features. Chiến lược 2 pha giúp head hội tụ trước rồi mới fine-tune toàn bộ, tránh phá pretrained features.

**Artifacts được tạo:**
- `artifacts/checkpoints/resnet50_best.pt`
- `artifacts/checkpoints/mobilenet_v2_best.pt`
- `artifacts/checkpoints/efficientnet_b0_best.pt`
- `artifacts/training/resnet50_history.json`
- `artifacts/training/mobilenet_v2_history.json`
- `artifacts/training/efficientnet_b0_history.json`

---

## Week 10 — Evaluation Report

### Tasks đã thực hiện

1. **Tái tạo kiến trúc model** — Xây lại AlexNet + 3 backbone transfer learning để load `state_dict` tương thích
2. **Hàm đánh giá `collect_predictions`** — Chạy inference trên toàn bộ loader, trả về `(y_true, y_pred)` cho sklearn
3. **Tính toàn bộ metrics** — Accuracy, Precision (macro/weighted), Recall (macro/weighted), F1-score (macro/weighted), Confusion matrix
4. **Đánh giá tất cả models trên 3 tập** — train, val, test cho mỗi model có checkpoint
5. **Bảng tổng hợp so sánh** — Ma trận model × split × 7 metrics
6. **Classification Report chi tiết** — `sklearn.classification_report` cho từng model trên test set (120 class)
7. **Confusion Matrix** — Full heatmap + Top-20 confused pairs (bar chart ngang)
8. **So sánh Accuracy bar chart** — Cả 4 models × 3 splits trên cùng figure
9. **So sánh F1-score (macro) bar chart** — Tương tự accuracy
10. **Per-class F1 distribution** — Boxplot per-class F1 cho từng model, thể hiện độ đồng đều qua 120 breeds
11. **Top-10 breeds khó nhất và dễ nhất** — Dựa trên F1 per-class của model tốt nhất
12. **Lưu kết quả ra JSON** — Export tất cả metrics (không kèm confusion matrix lớn)

### Output nhận được

**Bảng tổng hợp (trên test set):**

Mỗi model được đo trên cả 3 tập với 7 chỉ số:
- Accuracy (top-1)
- Precision macro / weighted
- Recall macro / weighted
- F1-score macro / weighted

**Phân tích Confusion Matrix:**
- Full confusion matrix 120×120 dạng heatmap cho mỗi model
- Top-20 cặp giống chó hay bị nhầm lẫn nhất (thường là các cặp cùng họ: terrier, spaniel, hound...)

**Per-class F1 Distribution:**
- Boxplot cho thấy mức độ đồng đều của model qua các breeds
- Mean, median, min, max F1 per-class cho từng model

**Top-10 Hardest/Easiest Breeds:**
- Breeds dễ nhất: F1 cao, thường là giống có ngoại hình đặc trưng rõ ràng
- Breeds khó nhất: F1 thấp, thường là giống tương đồng về kết cấu lông, hình dáng

**Artifacts được tạo:**
- `artifacts/training/week10_evaluation_results.json`

---

## Week 11 — Kiểm thử, tối ưu hệ thống và độ ổn định mô hình

### Tasks đã thực hiện

#### Phần 1: Kiểm thử dữ liệu (Data Integrity Tests)

1. **Kiểm tra file artifacts tồn tại** — class_to_idx.json, train/val/test_records.json
2. **Kiểm tra số class** — Phải đúng 120
3. **Kiểm tra label range** — min >= 0, max < num_classes cho mỗi split
4. **Kiểm tra class coverage** — Tất cả 120 class phải có mặt trong mỗi split
5. **Kiểm tra image/annotation paths tồn tại** — Không có file bị thiếu
6. **Thống kê class balance** — min, median, max samples per class cho mỗi split
7. **Kiểm tra split leakage** — Không có ảnh trùng giữa train/val/test
8. **Kiểm tra eval batch** — Shape đúng `(B, 3, 224, 224)`, không có NaN/Inf

#### Phần 2: Kiểm thử checkpoint (Smoke Tests)

9. **Load checkpoint và forward pass** — Mỗi model load state_dict thành công
10. **Kiểm tra output shape** — Phải là `(batch, 120)`
11. **Kiểm tra finite values** — Logits không chứa NaN/Inf

#### Phần 3: Benchmark hệ thống

12. **Benchmark DataLoader workers** — Test throughput (img/s) với num_workers = [0, 2, 4, 8, 12], tìm cấu hình tối ưu
13. **Benchmark inference throughput** — Đo img/s, ms/batch, peak GPU memory cho từng model

#### Phần 4: Đánh giá thực nghiệm

14. **Đánh giá val/test chi tiết** — Loss, Top-1, Top-5, Precision/Recall/F1 (macro/weighted), mean confidence
15. **Bảng tổng hợp** — Model × split × metrics
16. **Bar chart Top-1 và F1 macro** — val vs test cho tất cả models

#### Phần 5: Kiểm tra độ ổn định

17. **Repeated batch stability** — Cùng batch chạy 5 lần, so sánh prediction agreement và max absolute logit delta
18. **Repeated metric stability** — Cùng subset đánh giá chạy 3 lần, đo std của Top-1/Top-5/loss

#### Phần 6: Robustness

19. **Test-Time Augmentation (TTA)** — RandomResizedCrop(scale=0.9-1.0) + RandomHorizontalFlip, so sánh prediction agreement giữa crop cố định vs crop ngẫu nhiên
20. **Đo confidence shift** — Base confidence vs TTA confidence

#### Phần 7: Phân tích lỗi

21. **Per-class accuracy** — Từ confusion matrix, tính accuracy cho từng breed
22. **Top confused pairs** — 15 cặp breed hay nhầm lẫn nhất (count, true → pred)
23. **Hardest/Easiest classes** — Bar chart 10 breeds khó nhất vs 10 breeds dễ nhất
24. **Misclassified examples** — 12 ảnh dự đoán sai có confidence cao nhất (lỗi đáng phân tích nhất)

#### Phần 8: Xuất kết quả

25. **Classification report** — sklearn report cho model tốt nhất trên test set
26. **Export JSON** — Toàn bộ kết quả kiểm thử, benchmark, stability, TTA, metrics

### Output nhận được

**Data Integrity Tests:**

| Test | Status |
|------|--------|
| required_artifact_files | PASS |
| num_classes == 120 | PASS |
| train/val/test_label_range | PASS |
| train/val/test_class_coverage | PASS |
| train/val/test_image_paths_exist | PASS |
| train/val/test_annotation_paths_exist | PASS |
| split_leakage_by_image_path | PASS (không leak) |
| eval_batch_shape_and_finite_values | PASS |

**Checkpoint Smoke Tests:**
- Tất cả models có checkpoint đều load thành công
- Output shape đúng `(batch, 120)`, logits hữu hạn

**DataLoader Benchmark:**
- Đo throughput (img/s) với nhiều cấu hình num_workers
- Khuyến nghị cấu hình tối ưu dựa trên kết quả benchmark

**Inference Benchmark:**
- Đo img/s, ms/batch, peak GPU memory cho từng model
- Transfer learning models nặng hơn AlexNet nhưng throughput vẫn đủ nhanh

**Stability Tests:**
- Prediction agreement = 1.0000 (hoàn toàn ổn định khi chạy lặp)
- Max absolute logit delta ≈ 0 (trên CPU) hoặc rất nhỏ (trên GPU do floating point)
- Metric std ≈ 0 (Top-1, Top-5, loss không dao động)

**TTA Robustness:**
- Agreement giữa crop cố định và crop ngẫu nhiên: phản ánh mức độ robust
- Model mạnh hơn có agreement cao hơn

**Error Analysis:**
- Top confused pairs thường là cặp giống cùng họ (terrier, spaniel, hound)
- Ảnh dự đoán sai với confidence cao cho thấy model overconfident ở một số trường hợp edge

**Artifacts được tạo:**
- `artifacts/training/week11_system_test_results.json`

---

## Tổng kết tiến trình

| Tuần | Nội dung chính | Deliverable |
|------|----------------|-------------|
| Week 4 | Khám phá dữ liệu Stanford Dogs | Thống kê 120 classes, 20.580 ảnh, phân phối cân bằng |
| Week 5 | Tiền xử lý, manifest, split, augmentation, Dataset/DataLoader | Artifacts dữ liệu (class_to_idx, train/val/test records) |
| Week 6-7 | Implement & train AlexNet from scratch | Checkpoint baseline, training history |
| Week 8-9 | Transfer learning 3 backbone (2-phase strategy) | 3 checkpoints, training histories, bảng so sánh |
| Week 10 | Evaluation report toàn diện | Metrics 7 chỉ số × 4 models × 3 splits, confusion analysis |
| Week 11 | Kiểm thử, benchmark, stability, robustness, error analysis | JSON kết quả kiểm thử, khuyến nghị cải thiện |

### Cấu trúc artifacts

```
artifacts/
├── datasets/
│   ├── class_to_idx.json
│   ├── train_records.json
│   ├── val_records.json
│   └── test_records.json
├── checkpoints/
│   ├── alexnet_from_scratch_best.pt
│   ├── resnet50_best.pt
│   ├── mobilenet_v2_best.pt
│   └── efficientnet_b0_best.pt
└── training/
    ├── alexnet_from_scratch_history.json
    ├── resnet50_history.json
    ├── mobilenet_v2_history.json
    ├── efficientnet_b0_history.json
    ├── week10_evaluation_results.json
    └── week11_system_test_results.json
```
