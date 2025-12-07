"""
src/train_model.py

Unified training script for DeepFake detection models.
Supports training:
- Xception model (for images)
- Hybrid model (Xception + EfficientNetB4 + ResNet50 for images)
- Video sequence model (CNN-RNN for videos)

Usage:
    # Train Xception model
    python src/train_model.py --model xception
    
    # Train Hybrid model
    python src/train_model.py --model hybrid
    
    # Train Video model
    python src/train_model.py --model video
"""

import os
import sys
import argparse

# Ensure local src/ package is importable even if a global 'src' package exists
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import model as model_module

def train_xception_model():
    """Train Xception-based image model."""
    print("=" * 70)
    print("DeepFake Detection Model Training")
    print("Using Xception Transfer Learning for 90+ Accuracy")
    print("=" * 70)
    
    # Configuration
    data_folder = "data/image_data"
    sample_size = 16000  # 8000 per class
    batch_size = 32
    initial_epochs = 5  # Epochs with frozen base
    fine_tune_epochs = 10  # Epochs for fine-tuning
    checkpoint_path = "xception_deepfake_model.h5"
    
    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset from folder...")
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = model_module.load_dataset_from_folder(
            data_folder=data_folder,
            sample_size=sample_size,
            random_state=42
        )
        print(f"✓ Dataset loaded successfully!")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Build Xception model
    print("\n[Step 2] Building Xception model with transfer learning...")
    try:
        model, base_model = model_module.build_xception_model(
            input_shape=(224, 224, 3),
            num_classes=1,
            use_binary=True
        )
        print(f"✓ Model built successfully!")
        print(f"  Model name: {model.name}")
        print(f"  Total parameters: {model.count_params():,}")
        model.summary()
    except Exception as e:
        print(f"✗ Error building model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Train model
    print("\n[Step 3] Training model...")
    try:
        history1, history2, trained_model = model_module.train_model_with_dataset(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=initial_epochs,
            batch_size=batch_size,
            use_callbacks=True,
            checkpoint_path=checkpoint_path,
            fine_tune_epochs=fine_tune_epochs,
            unfreeze_from_layer=56,
            base_model=base_model
        )
        print(f"✓ Training completed!")
        
        # Update model reference
        model = trained_model
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Evaluate on test set
    print("\n[Step 4] Evaluating on test set...")
    try:
        test_dataset = model_module.prepare_tf_dataset(
            X_test, y_test, batch_size=batch_size, shuffle=False, use_xception_preprocess=True
        )
        test_results = model.evaluate(test_dataset, verbose=1)
        print(f"✓ Test evaluation completed!")
        print(f"  Test Loss: {test_results[0]:.4f}")
        print(f"  Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        
        if test_results[1] >= 0.90:
            print(f"\n🎉 SUCCESS! Model achieved 90+ accuracy: {test_results[1]*100:.2f}%")
        else:
            print(f"\n⚠ Model accuracy is {test_results[1]*100:.2f}%. Consider:")
            print(f"  - Training for more epochs")
            print(f"  - Increasing sample_size")
            print(f"  - Adjusting learning rates")
            
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Save final model
    print(f"\n[Step 5] Saving final model to {checkpoint_path}...")
    try:
        model.save(checkpoint_path)
        print(f"✓ Model saved successfully!")
        print(f"\n📁 Model saved at: {os.path.abspath(checkpoint_path)}")
        print(f"\nTo use this model in deployment, set MODEL_PATH environment variable:")
        print(f"  export MODEL_PATH={os.path.abspath(checkpoint_path)}")
        print(f"  Or update app/main.py to use: MODEL_PATH = '{checkpoint_path}'")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("Training pipeline completed successfully!")
    print("=" * 70)

def train_hybrid_model():
    """Train Hybrid model (Xception + EfficientNetB4 + ResNet50)."""
    print("=" * 70)
    print("DeepFake Detection - HYBRID MODEL Training")
    print("Combining Xception + EfficientNetB4 + ResNet50")
    print("Target: 99%+ Accuracy")
    print("=" * 70)
    
    # Configuration
    data_folder = "data/image_data"
    sample_size = 8000  # Reduced to avoid memory issues (4000 per class) - can increase if more RAM available
    batch_size = 8  # Smaller batch size to reduce memory usage
    initial_epochs = 8  # More epochs with frozen base
    fine_tune_epochs = 15  # More epochs for fine-tuning
    checkpoint_path = "hybrid_deepfake_model.h5"
    
    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset from folder...")
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = model_module.load_dataset_from_folder(
            data_folder=data_folder,
            sample_size=sample_size,
            random_state=42
        )
        print(f"✓ Dataset loaded successfully!")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Check for existing checkpoint and load or build model
    print("\n[Step 2] Checking for existing checkpoint...")
    checkpoint_exists = os.path.exists(checkpoint_path)
    
    if checkpoint_exists:
        print(f"✓ Found existing checkpoint: {checkpoint_path}")
        print("  Loading model from checkpoint to resume training...")
        try:
            model = model_module.load_model_from_checkpoint(checkpoint_path)
            print(f"✓ Model loaded successfully from checkpoint!")
            print(f"  Model name: {model.name}")
            print(f"  Total parameters: {model.count_params():,}")
            
            # Rebuild base_models_dict for fine-tuning (needed for unfreeze)
            print("  Rebuilding base models for fine-tuning...")
            _, base_models_dict = model_module.build_hybrid_model(
                input_shape=(224, 224, 3),
                num_classes=1,
                use_binary=True
            )
            print("✓ Ready to resume training from checkpoint!")
            resume_training = True
        except Exception as e:
            print(f"⚠ Warning: Could not load checkpoint ({e})")
            print("  Building new model instead...")
            resume_training = False
            model, base_models_dict = model_module.build_hybrid_model(
                input_shape=(224, 224, 3),
                num_classes=1,
                use_binary=True
            )
            print(f"✓ Hybrid model built successfully!")
    else:
        print("  No checkpoint found. Building new model...")
        resume_training = False
        try:
            model, base_models_dict = model_module.build_hybrid_model(
                input_shape=(224, 224, 3),
                num_classes=1,
                use_binary=True
            )
            print(f"✓ Hybrid model built successfully!")
            print(f"  Model name: {model.name}")
            print(f"  Total parameters: {model.count_params():,}")
            print(f"  Base models: {list(base_models_dict.keys())}")
            print("\nModel Architecture Summary:")
            model.summary()
        except Exception as e:
            print(f"✗ Error building hybrid model: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 3: Train model
    print("\n[Step 3] Training hybrid model...")
    if resume_training:
        print("  ⚠ Resuming training from checkpoint!")
        print("  Note: Will skip Phase 1 and proceed directly to Phase 2 (Fine-tuning)")
        print("  If you want to restart from Phase 1, delete the checkpoint file first.")
    
    try:
        history1, history2, trained_model = model_module.train_model_with_dataset(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=initial_epochs if not resume_training else 0,  # Skip Phase 1 if resuming
            batch_size=batch_size,
            use_callbacks=True,
            checkpoint_path=checkpoint_path,
            fine_tune_epochs=fine_tune_epochs,
            unfreeze_from_layer=100,
            base_models_dict=base_models_dict,
            resume_from_checkpoint=resume_training
        )
        print(f"✓ Training completed!")
        
        # Update model reference
        model = trained_model
        
        # Print training history
        if history1:
            print(f"\nPhase 1 (Frozen Base) - Final Accuracy: {max(history1.history.get('val_accuracy', history1.history.get('accuracy', [0]))):.4f}")
        if history2:
            print(f"Phase 2 (Fine-tuning) - Final Accuracy: {max(history2.history.get('val_accuracy', history2.history.get('accuracy', [0]))):.4f}")
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Evaluate on test set
    print("\n[Step 4] Evaluating hybrid model on test set...")
    try:
        test_dataset = model_module.prepare_tf_dataset(
            X_test, y_test, 
            batch_size=batch_size, 
            shuffle=False, 
            use_hybrid=True
        )
        test_results = model.evaluate(test_dataset, verbose=1)
        print(f"✓ Test evaluation completed!")
        print(f"  Test Loss: {test_results[0]:.4f}")
        print(f"  Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        
        if len(test_results) > 2:
            print(f"  Test Precision: {test_results[2]:.4f}")
            print(f"  Test Recall: {test_results[3]:.4f}")
        
        if test_results[1] >= 0.99:
            print(f"\n🎉🎉🎉 EXCELLENT! Hybrid model achieved 99%+ accuracy: {test_results[1]*100:.2f}% 🎉🎉🎉")
        elif test_results[1] >= 0.95:
            print(f"\n🎉 GREAT! Hybrid model achieved 95%+ accuracy: {test_results[1]*100:.2f}%")
            print(f"   Consider training for more epochs or increasing sample_size for 99%+")
        elif test_results[1] >= 0.90:
            print(f"\n✓ Good! Model accuracy is {test_results[1]*100:.2f}%")
            print(f"   To reach 99%+, try:")
            print(f"   - Training for more epochs")
            print(f"   - Increasing sample_size (currently {sample_size})")
            print(f"   - Adjusting learning rates")
        else:
            print(f"\n⚠ Model accuracy is {test_results[1]*100:.2f}%")
            print(f"   Consider:")
            print(f"   - Training for more epochs")
            print(f"   - Increasing sample_size")
            print(f"   - Checking data quality")
            
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Save final model
    print(f"\n[Step 5] Saving hybrid model to {checkpoint_path}...")
    try:
        model.save(checkpoint_path)
        print(f"✓ Hybrid model saved successfully!")
        print(f"\n📁 Model saved at: {os.path.abspath(checkpoint_path)}")
        print(f"\nTo use this model in deployment, set MODEL_PATH environment variable:")
        print(f"  export MODEL_PATH={os.path.abspath(checkpoint_path)}")
        print(f"  Or update app/main.py to use: MODEL_PATH = '{checkpoint_path}'")
        print(f"\n💡 The hybrid model will automatically be detected and used correctly!")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("Hybrid model training pipeline completed successfully!")
    print("=" * 70)
    print("\n📊 Model Performance Summary:")
    print(f"   - Architecture: Xception + EfficientNetB4 + ResNet50")
    print(f"   - Parameters: {model.count_params():,}")
    print(f"   - Test Accuracy: {test_results[1]*100:.2f}%")
    print(f"   - Model saved: {checkpoint_path}")
    print("\n🚀 Your hybrid model is ready for deployment!")

def train_video_model():
    """Train CNN-RNN video sequence model."""
    print("=" * 70)
    print("DeepFake Video Detection Model Training")
    print("Using CNN-RNN Architecture for Video Classification")
    print("=" * 70)
    
    # Configuration
    data_folder = "data/videos_data/train_sample_videos"
    metadata_file = "metadata.json"
    sample_size = 200  # Number of videos to use (None = all)
    batch_size = 8
    epochs = 10
    max_seq_length = 20
    num_features = 2048  # InceptionV3 output features
    img_size = 224
    checkpoint_path = "video_deepfake_model.h5"
    
    # Step 1: Load video dataset
    print("\n[Step 1] Loading video dataset from folder...")
    try:
        (train_paths, y_train), (val_paths, y_val), (test_paths, y_test) = model_module.load_video_dataset_from_folder(
            data_folder=data_folder,
            metadata_file=metadata_file,
            sample_size=sample_size,
            random_state=42
        )
        print(f"✓ Dataset loaded successfully!")
        print(f"  Training: {len(train_paths)} videos")
        print(f"  Validation: {len(val_paths)} videos")
        print(f"  Test: {len(test_paths)} videos")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Build feature extractor
    print("\n[Step 2] Building InceptionV3 feature extractor...")
    try:
        feature_extractor = model_module.build_video_feature_extractor(
            input_shape=(img_size, img_size, 3)
        )
        print(f"✓ Feature extractor built successfully!")
        print(f"  Output features: {feature_extractor.output_shape[-1]}")
    except Exception as e:
        print(f"✗ Error building feature extractor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Extract features from videos
    print("\n[Step 3] Extracting features from training videos...")
    print("  This may take a while...")
    try:
        train_data, train_labels = model_module.prepare_all_videos_for_training(
            train_paths,
            y_train,
            feature_extractor,
            max_seq_length=max_seq_length,
            img_size=img_size
        )
        print(f"✓ Training features extracted!")
        print(f"  Frame features shape: {train_data[0].shape}")
        print(f"  Frame masks shape: {train_data[1].shape}")
    except Exception as e:
        print(f"✗ Error extracting training features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[Step 4] Extracting features from validation videos...")
    try:
        val_data, val_labels = model_module.prepare_all_videos_for_training(
            val_paths,
            y_val,
            feature_extractor,
            max_seq_length=max_seq_length,
            img_size=img_size
        )
        print(f"✓ Validation features extracted!")
    except Exception as e:
        print(f"✗ Error extracting validation features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Build video sequence model
    print("\n[Step 5] Building CNN-RNN video sequence model...")
    try:
        video_model = model_module.build_video_sequence_model(
            max_seq_length=max_seq_length,
            num_features=num_features,
            num_classes=1,
            use_binary=True
        )
        print(f"✓ Video model built successfully!")
        video_model.summary()
    except Exception as e:
        print(f"✗ Error building video model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Train model
    print("\n[Step 6] Training video model...")
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = video_model.fit(
            [train_data[0], train_data[1]],
            train_labels,
            validation_data=([val_data[0], val_data[1]], val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✓ Training completed!")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Evaluate on test set
    print("\n[Step 7] Evaluating on test set...")
    try:
        print("  Extracting test features...")
        test_data, test_labels = model_module.prepare_all_videos_for_training(
            test_paths,
            y_test,
            feature_extractor,
            max_seq_length=max_seq_length,
            img_size=img_size
        )
        
        print("  Evaluating model...")
        test_results = video_model.evaluate(
            [test_data[0], test_data[1]],
            test_labels,
            verbose=1
        )
        
        print(f"✓ Test evaluation completed!")
        print(f"  Test Loss: {test_results[0]:.4f}")
        print(f"  Test Accuracy: {test_results[1]:.4f}")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 7: Save model
    print(f"\n[Step 8] Saving model to {checkpoint_path}...")
    try:
        video_model.save(checkpoint_path)
        print(f"✓ Model saved successfully!")
        print(f"\n📁 Model saved at: {os.path.abspath(checkpoint_path)}")
        print(f"\nTo use this model in deployment, set VIDEO_MODEL_PATH environment variable:")
        print(f"  export VIDEO_MODEL_PATH={os.path.abspath(checkpoint_path)}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print(f"Model saved to: {checkpoint_path}")
    print("=" * 70)

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train DeepFake detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/train_model.py --model xception    # Train Xception image model
  python src/train_model.py --model hybrid      # Train Hybrid image model
  python src/train_model.py --model video       # Train Video sequence model
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['xception', 'hybrid', 'video'],
        default='xception',
        help='Model type to train (default: xception)'
    )
    
    args = parser.parse_args()
    
    if args.model == 'xception':
        train_xception_model()
    elif args.model == 'hybrid':
        train_hybrid_model()
    elif args.model == 'video':
        train_video_model()
    else:
        print(f"Unknown model type: {args.model}")
        print("Available options: xception, hybrid, video")
        sys.exit(1)

if __name__ == "__main__":
    main()

