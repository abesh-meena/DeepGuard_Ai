# DeepGuard AI - DeepFake Detection System

A powerful DeepFake detection system using hybrid deep learning models combining Xception, EfficientNetB4, and ResNet50 architectures. This project provides both training capabilities and a production-ready deployment API.

## 🎯 Features

- **Hybrid Model Architecture**: Combines Xception, EfficientNetB4, and ResNet50 for 99%+ accuracy
- **Single Model Option**: Xception-based model for 90%+ accuracy
- **Complete Training Pipeline**: Automated training with data augmentation and fine-tuning
- **Production API**: FastAPI backend for model inference
- **User-Friendly UI**: Streamlit interface for easy testing
- **Dataset Management**: Automatic dataset loading and preprocessing

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow 2.10+
- CUDA-compatible GPU (recommended for training, optional for inference)

## 🚀 Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd DeepGuard_AI
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
DeepGuard_AI/
├── app/
│   └── main.py               # FastAPI backend (image prediction API)
├── src/
│   ├── model.py              # Core model functions (Xception, Hybrid, video models + training helpers)
│   ├── utils.py              # Helper utilities for images/videos
│   └── __init__.py
├── ui/
│   └── streamlit_app.py      # Streamlit UI (talks to FastAPI backend)
├── data/
│   ├── image_data/           # Image dataset folder
│   │   ├── metadata.csv      # Dataset metadata
│   │   └── Afaces_224/       # Image files
│   └── videos_data/          # (Optional) Video dataset for video model
├── ml_model.ipynb            # Notebook (experiments / exploration)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🎓 Training Models

All training options are handled via a single script: `src/train_model.py`.

1. **Train Xception Model (90%+ Accuracy)**

```bash
python src/train_model.py --model xception
```

**Default configuration** (inside `src/train_model.py`):
- Sample size: 16,000 images (8k per class)
- Batch size: 32
- Initial epochs: 5 (frozen base)
- Fine-tuning epochs: 10
- Output: `xception_deepfake_model.h5`

2. **Train Hybrid Model (Xception + EfficientNetB4 + ResNet50, 99%+ Accuracy)**

```bash
python src/train_model.py --model hybrid
```

**Default configuration**:
- Sample size: 20,000 images (10k per class)
- Batch size: 16
- Initial epochs: 8 (frozen base)
- Fine-tuning epochs: 15
- Output: `hybrid_deepfake_model.h5`

3. **Train Video Model (CNN‑RNN)**

```bash
python src/train_model.py --model video
```

**Default configuration**:
- Dataset: `data/videos_data/train_sample_videos` with `metadata.json`
- Max frames per video: 20
- Features: InceptionV3 feature extractor
- Output: `video_deepfake_model.h5`

### Training Process

Both scripts follow a two-phase training approach:

1. **Phase 1**: Train with frozen base models (transfer learning)
2. **Phase 2**: Fine-tune top layers with lower learning rate

The training includes:
- Data augmentation (RandomFlip, RandomRotation, RandomContrast)
- Early stopping
- Learning rate reduction
- Model checkpointing

## 🚀 Deployment

### 1. Start FastAPI Backend

```bash
# Set model path (optional, defaults to "model_checkpoint")
export MODEL_PATH=hybrid_deepfake_model.h5

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or run directly:
```bash
python app/main.py
```

### 2. Start Streamlit UI (Optional)

In a new terminal:

```bash
streamlit run ui/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### API Endpoints

#### Health Check
```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### Prediction
```bash
POST http://localhost:8000/predict
Content-Type: multipart/form-data

# Upload image file
```

Response:
```json
{
  "prediction": "fake",
  "probabilities": [[0.1, 0.9]]
}
```

## 📊 Model Architectures

### Hybrid Model

Combines three powerful architectures:

- **Xception**: Excellent feature extraction
- **EfficientNetB4**: Efficient and powerful
- **ResNet50**: Strong residual learning

**Architecture Flow**:
1. Input (224×224×3)
2. Data Augmentation
3. Three parallel branches (Xception, EfficientNet, ResNet)
4. Feature Fusion (Concatenation)
5. Deep Fusion Layers (1024 → 512 → 256)
6. Output (Binary: Real/Fake)

**Total Parameters**: ~50-60 million

### Xception Model

Single architecture model:

- **Base**: Xception (ImageNet pretrained)
- **Features**: Global Average Pooling
- **Classifier**: Dense layers (256 → 128)
- **Output**: Binary classification

**Total Parameters**: ~20 million

## 🔧 Configuration

### Dataset Setup

Place your dataset in the following structure:

```
data/
└── image_data/
    ├── metadata.csv          # CSV with columns: videoname, label
    └── Afaces_224/           # Folder with .jpg images
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

**metadata.csv format**:
```csv
videoname,original_width,original_height,label,original
image1.mp4,224,224,REAL,
image2.mp4,224,224,FAKE,original.mp4
```

### Model Path Configuration

Set the model path in `app/main.py` or via environment variable:

```python
MODEL_PATH = "hybrid_deepfake_model.h5"  # or "xception_deepfake_model.h5"
```

Or use environment variable:
```bash
export MODEL_PATH=hybrid_deepfake_model.h5
```

## 📈 Performance

### Hybrid Model
- **Target Accuracy**: 99%+
- **Training Time**: ~2-4 hours (GPU recommended)
- **Inference Time**: ~50-100ms per image (GPU)
- **Model Size**: ~200-300 MB

### Xception Model
- **Target Accuracy**: 90%+
- **Training Time**: ~1-2 hours (GPU recommended)
- **Inference Time**: ~20-50ms per image (GPU)
- **Model Size**: ~80-100 MB

## 🛠️ Usage Examples

### Python API Usage

```python
from src import model as model_module
import numpy as np
from PIL import Image

# Load model
model = model_module.load_model_from_checkpoint("hybrid_deepfake_model.h5")

# Load and preprocess image
img = Image.open("test_image.jpg").convert("RGB")
arr = np.asarray(img.resize((224, 224)))

# Predict
result = model_module.predict_from_input(model, arr)
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: {result['probabilities']}")
```

### Training Custom Dataset

```python
from src import model as model_module

# Load dataset
(X_train, y_train), (X_val, y_val), (X_test, y_test) = model_module.load_dataset_from_folder(
    data_folder="data/image_data",
    sample_size=20000,
    random_state=42
)

# Build hybrid model
model, base_models_dict = model_module.build_hybrid_model(
    input_shape=(224, 224, 3),
    num_classes=1,
    use_binary=True
)

# Train
history1, history2, trained_model = model_module.train_model_with_dataset(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=8,
    batch_size=16,
    base_models_dict=base_models_dict
)

# Save
trained_model.save("my_custom_model.h5")
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in training script
   - Use smaller sample_size
   - Enable mixed precision training

2. **Model Not Found**
   - Check MODEL_PATH environment variable
   - Ensure model file exists in project directory
   - Verify file extension (.h5)

3. **Import Errors**
   - Ensure virtual environment is activated
   - Run: `pip install -r requirements.txt`
   - Check Python version (3.8+)

4. **CUDA/GPU Issues**
   - Install CUDA-compatible TensorFlow: `pip install tensorflow-gpu`
   - Verify GPU availability: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## 📝 Notes

- **GPU Recommended**: Training is significantly faster on GPU
- **Memory Requirements**: Hybrid model requires ~8GB+ GPU memory
- **Dataset Size**: Larger datasets (20k+ samples) yield better accuracy
- **Training Time**: Be patient, fine-tuning can take several hours
- **Model Compatibility**: All models are backward compatible with existing deployment code

## 🔮 Future Enhancements

- [ ] Video frame sequence analysis
- [ ] Real-time webcam detection
- [ ] Model quantization for mobile deployment
- [ ] REST API authentication
- [ ] Batch prediction endpoint
- [ ] Model versioning system

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- TensorFlow/Keras team for excellent deep learning framework
- ImageNet dataset for pretrained models
- DeepFake detection challenge datasets

## 📧 Support

For issues or questions, please check the code comments in `src/model.py` or create an issue in the repository.

---

**Happy Training! 🚀**
