import os
import sys
import argparse
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deepface_verification")

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def install_deepface():
    """Install DeepFace if not already installed"""
    try:
        import deepface
        logger.info(f"DeepFace already installed (version: {deepface.__version__})")
        return True
    except ImportError:
        logger.info("DeepFace not found. Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "deepface"])
            logger.info("DeepFace installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install DeepFace: {e}")
            return False

def verify_faces(img1_path, img2_path, model_name="ArcFace", detector_backend="retinaface", 
                distance_metric="cosine", enforce_detection=False, visualize=True):
    """
    Verify if two face images belong to the same person using DeepFace
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        model_name: Name of the face recognition model to use
        detector_backend: Face detection backend to use
        distance_metric: Distance metric for comparison
        enforce_detection: Whether to enforce face detection
        visualize: Whether to create visualization of results
        
    Returns:
        Dictionary with verification results
    """
    try:
        from deepface import DeepFace
        
        # Start timing
        start_time = time.time()
        
        # Available models
        available_models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        
        # Check if model is valid
        if model_name not in available_models:
            logger.warning(f"Invalid model name: {model_name}. Using ArcFace instead.")
            model_name = "ArcFace"
        
        # Verify faces
        logger.info(f"Verifying faces using {model_name} model and {detector_backend} detector...")
        
        # Read images to verify they exist and are valid
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None:
            raise ValueError(f"Could not read first image: {img1_path}")
        if img2 is None:
            raise ValueError(f"Could not read second image: {img2_path}")
        
        # Save originals to output directory
        cv2.imwrite(str(OUTPUT_DIR / "original_img1.jpg"), img1)
        cv2.imwrite(str(OUTPUT_DIR / "original_img2.jpg"), img2)
        
        # If not enforce_detection, try with enforce_detection=False first
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=enforce_detection
            )
        except Exception as e:
            if "Face could not be detected" in str(e) and not enforce_detection:
                logger.warning(f"Face detection failed. Retrying with enforce_detection=False")
                result = DeepFace.verify(
                    img1_path=img1_path,
                    img2_path=img2_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    enforce_detection=False
                )
            else:
                raise e
        
        # Extract verification result
        is_match = result.get('verified', False)
        distance = result.get('distance', 0)
        threshold = result.get('threshold', 0)
        model = result.get('model', model_name)
        
        # Calculate confidence level
        # Convert distance to similarity score (1 - distance) for better interpretability
        # Note: for cosine distance, lower is better (more similar)
        similarity = 1 - distance if distance_metric.lower() == "cosine" else 1 - distance/2
        
        # Get confidence level
        if distance_metric.lower() == "cosine":
            if distance <= threshold - 0.15:
                confidence = "very high"
            elif distance <= threshold - 0.05:
                confidence = "high"
            elif distance <= threshold:
                confidence = "medium"
            elif distance <= threshold + 0.05:
                confidence = "low"
            else:
                confidence = "very low"
        else:
            # For other distance metrics (e.g. euclidean)
            normalized_dist = distance / threshold
            if normalized_dist <= 0.5:
                confidence = "very high"
            elif normalized_dist <= 0.8:
                confidence = "high"
            elif normalized_dist <= 1.0:
                confidence = "medium"
            elif normalized_dist <= 1.2:
                confidence = "low"
            else:
                confidence = "very low"
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Try to get detected face regions if available
        try:
            # Extract face regions using DeepFace's extract faces
            faces_img1 = DeepFace.extract_faces(
                img_path=img1_path, 
                detector_backend=detector_backend,
                enforce_detection=enforce_detection
            )
            
            faces_img2 = DeepFace.extract_faces(
                img_path=img2_path,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection
            )
            
            # Save extracted faces if available
            if faces_img1 and len(faces_img1) > 0:
                face_img1 = faces_img1[0]['face'] * 255  # Convert from float [0,1] to uint8 [0,255]
                face_img1 = face_img1.astype(np.uint8)
                cv2.imwrite(str(OUTPUT_DIR / "face1.jpg"), cv2.cvtColor(face_img1, cv2.COLOR_RGB2BGR))
            
            if faces_img2 and len(faces_img2) > 0:
                face_img2 = faces_img2[0]['face'] * 255  # Convert from float [0,1] to uint8 [0,255]
                face_img2 = face_img2.astype(np.uint8)
                cv2.imwrite(str(OUTPUT_DIR / "face2.jpg"), cv2.cvtColor(face_img2, cv2.COLOR_RGB2BGR))
            
            region_available = True
        except Exception as e:
            logger.warning(f"Could not extract face regions: {e}")
            region_available = False
        
        # Create visualization
        if visualize:
            try:
                create_visualization(
                    img1_path, img2_path, 
                    is_match, similarity, threshold, 
                    confidence, model_name
                )
            except Exception as e:
                logger.warning(f"Could not create visualization: {e}")
        
        # Prepare detailed result
        detailed_result = {
            'match': is_match,
            'distance': float(distance),
            'threshold': float(threshold),
            'similarity': float(similarity),
            'confidence': confidence,
            'model': model,
            'detector': detector_backend,
            'distance_metric': distance_metric,
            'processing_time': float(processing_time),
            'message': f"Match {'found' if is_match else 'not found'} with {confidence} confidence ({similarity:.4f})"
        }
        
        return detailed_result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'match': False,
            'similarity': 0.0,
            'error': str(e),
            'message': f"Error during verification: {str(e)}"
        }

def create_visualization(img1_path, img2_path, is_match, similarity, threshold, confidence, model_name):
    """Create visualization of verification results"""
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        logger.warning("Could not read images for visualization")
        return
    
    # Convert to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Plot images
    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.title("Image 1")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2_rgb)
    plt.title("Image 2")
    plt.axis('off')
    
    # Add match/no match text
    match_color = 'green' if is_match else 'red'
    match_text = "MATCH" if is_match else "NO MATCH"
    
    plt.figtext(0.5, 0.01, match_text, ha='center', color=match_color, fontsize=20, weight='bold')
    
    # Add details text
    details = f"Model: {model_name}\nSimilarity: {similarity:.4f}\nThreshold: {threshold:.4f}\nConfidence: {confidence}"
    plt.figtext(0.5, 0.1, details, ha='center', fontsize=12)
    
    # Adjust spacing
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Save figure
    plt.savefig(str(OUTPUT_DIR / "verification_result.png"))
    logger.info(f"Visualization saved to {OUTPUT_DIR}/verification_result.png")

def analyze_with_multiple_models(img1_path, img2_path, detector_backend="retinaface", visualize=True):
    """
    Analyze two face images with multiple face recognition models
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        detector_backend: Face detection backend to use
        visualize: Whether to create visualization of results
        
    Returns:
        Dictionary with verification results for multiple models
    """
    from deepface import DeepFace
    
    # Available models
    models = ["VGG-Face", "Facenet", "OpenFace", "ArcFace", "DeepFace"]
    
    # Store results
    results = {}
    
    for model in models:
        try:
            logger.info(f"Analyzing with {model}...")
            result = verify_faces(
                img1_path, img2_path, 
                model_name=model, 
                detector_backend=detector_backend,
                visualize=False
            )
            results[model] = result
        except Exception as e:
            logger.warning(f"Error with {model}: {e}")
            results[model] = {
                'match': False,
                'error': str(e),
                'message': f"Error with {model}: {str(e)}"
            }
    
    # Find most common result
    match_count = sum(1 for model, result in results.items() if result.get('match', False))
    no_match_count = len(results) - match_count
    
    consensus = match_count > no_match_count
    confidence = "high" if abs(match_count - no_match_count) >= 3 else "medium" if abs(match_count - no_match_count) >= 2 else "low"
    
    # Create visualization with consensus result
    if visualize:
        try:
            # Use ArcFace result for visualization values (or first available)
            viz_model = results.get('ArcFace', next(iter(results.values())))
            similarity = viz_model.get('similarity', 0)
            threshold = viz_model.get('threshold', 0)
            
            create_visualization(
                img1_path, img2_path, 
                consensus, similarity, threshold, 
                confidence, "Ensemble"
            )
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")
    
    # Prepare consensus result
    consensus_result = {
        'match': consensus,
        'confidence': confidence,
        'match_count': match_count,
        'no_match_count': no_match_count,
        'model_results': results,
        'message': f"Match {'found' if consensus else 'not found'} with {confidence} confidence (voted by {match_count}/{len(results)} models)"
    }
    
    return consensus_result

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DeepFace Face Verification")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument("--model", choices=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "ensemble"],
                      default="ArcFace", help="Face recognition model to use")
    parser.add_argument("--detector", choices=["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"],
                      default="retinaface", help="Face detector backend")
    parser.add_argument("--metric", choices=["cosine", "euclidean", "euclidean_l2"],
                      default="cosine", help="Distance metric")
    parser.add_argument("--no-enforce-detection", action="store_true", 
                      help="Don't enforce face detection")
    parser.add_argument("--no-visualize", action="store_true",
                      help="Don't create visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Check if DeepFace is installed
    if not install_deepface():
        print("Error: DeepFace could not be installed. Please install it manually with 'pip install deepface'")
        return
    
    # Import DeepFace here after installation
    try:
        from deepface import DeepFace
        print(f"DeepFace version: {DeepFace.__version__}")
    except Exception as e:
        print(f"Error importing DeepFace: {e}")
        return
    
    # Print OpenCV version
    print(f"OpenCV version: {cv2.__version__}")
    
    # Process images
    if args.model.lower() == "ensemble":
        result = analyze_with_multiple_models(
            args.image1, args.image2,
            detector_backend=args.detector,
            visualize=not args.no_visualize
        )
    else:
        result = verify_faces(
            args.image1, args.image2,
            model_name=args.model,
            detector_backend=args.detector,
            distance_metric=args.metric,
            enforce_detection=not args.no_enforce_detection,
            visualize=not args.no_visualize
        )
    
    # Print result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()