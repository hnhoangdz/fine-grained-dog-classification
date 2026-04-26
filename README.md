# Fine-Grained Dog Classification

Project thực hành fine-grained image classification trên bộ Stanford Dogs. Mục tiêu chính là giúp sinh viên nắm được pipeline nền tảng: lấy dữ liệu, preprocess, train baseline, transfer learning, evaluation, kiểm thử độ ổn định và tổng hợp kết quả thực nghiệm.

## Yêu Cầu

- Python >= 3.12
- `uv` để quản lý môi trường
- NVIDIA GPU được khuyến nghị cho các notebook training

Khởi tạo môi trường:

```bash
uv sync
```

Mở Jupyter từ môi trường của project:

```bash
uv run jupyter lab
```

Nếu dùng VS Code/Jupyter, chọn kernel từ `.venv` của repo để đảm bảo có `torch`/`torchvision`.

## Thứ Tự Chạy Notebook

Chạy theo thứ tự này nếu bắt đầu từ đầu:

1. `week4_datasets.ipynb`
   Tải và kiểm tra dữ liệu Stanford Dogs.

2. `week5_preprocess-datasets.ipynb`
   Tạo manifest, chia `train`/`val`/`test`, và lưu metadata vào `artifacts/datasets`.

3. `week67_alexnet-from-scratch.ipynb`
   Train baseline AlexNet from scratch. Notebook này là bản chính cho Week 6-7.

4. `week89_transfer_learning.ipynb`
   Train các model transfer learning: ResNet50, MobileNetV2, EfficientNet-B0.

5. `week10_evaluation_report.ipynb`
   Đánh giá và so sánh các model đã train bằng accuracy, F1, classification report và confusion analysis.

6. `week11_testing_optimization_stability.ipynb`
   Kiểm thử dữ liệu/checkpoint, benchmark DataLoader/inference, kiểm tra độ ổn định model, robustness nhẹ và xuất kết quả thực nghiệm.

## Artifacts

Notebook sẽ đọc/ghi các file trong `artifacts/`:

- `artifacts/datasets/class_to_idx.json`
- `artifacts/datasets/train_records.json`
- `artifacts/datasets/val_records.json`
- `artifacts/datasets/test_records.json`
- `artifacts/checkpoints/alexnet_from_scratch_best.pt`
- `artifacts/checkpoints/resnet50_best.pt`
- `artifacts/checkpoints/mobilenet_v2_best.pt`
- `artifacts/checkpoints/efficientnet_b0_best.pt`
- `artifacts/training/*_history.json`
- `artifacts/training/week10_evaluation_results.json`
- `artifacts/training/week11_system_test_results.json`

Nếu checkpoint transfer learning chưa tồn tại, Week 10 và Week 11 vẫn chạy được và sẽ skip model tương ứng.

## Cấu Hình Hiện Tại

- Image size: `224x224`
- Batch size training: `32`
- DataLoader workers: `8`
- Transfer learning dùng ImageNet normalization
- Week 11 dùng evaluation deterministic transform: `Resize(256) -> CenterCrop(224) -> Normalize`

Trên GPU 8 GiB, cấu hình hiện tại đủ VRAM cho các model trong project. `NUM_WORKERS=8` giúp giảm nghẽn DataLoader so với `NUM_WORKERS=0`.

## Ghi Chú

- `week67_alexnet-from-scratch.ipynb` thay thế file `week6_alexnet-from-scratch.ipynb` cũ.
- Không cần train lại Week 4-5 nếu các file trong `artifacts/datasets` đã tồn tại và không thay đổi dữ liệu.
- Nếu muốn so sánh đầy đủ trong Week 10/11, hãy train xong cả AlexNet và các model transfer learning trước.
