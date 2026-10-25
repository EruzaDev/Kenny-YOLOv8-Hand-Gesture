# Thumbs-up evaluation

The app returns only thumbs-up detections. The shown percentage is model confidence, not measured accuracy. Open the [interactive image viewer](evaluation_viewer.html) for labeled and predicted boxes on all 99 evaluated images.

At the app's 0.80 confidence threshold and IoU ≥ 0.50:

| Set | Labeled thumbs-up boxes | Correct | False detections | Missed | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Photo validation | 26 | 26 | 0 | 0 | 100% | 100% | 100% |
| Webcam crops | 19 | 17 | 0 | 2 | 100% | 89.5% | 94.4% |

The 11 negative photo-validation images and 50 training negatives produced no detections at 0.80. This is a small evaluation: photo labels are teacher-generated, webcam crops share source images with photo validation, and training negatives are not independent. See [uncertainty intervals](evaluation_uncertainty.png) and [per-image results](per_image_results.csv).

## Graphs and visualizers

- [Final thumbs-up performance](thumbs_up_performance.png) and [detection outcomes](detection_outcomes.png)
- [Detection examples](detection_examples.png), with both missed crops first
- [Threshold tradeoff](threshold_optimization.png) and [underlying CSV](threshold_optimization.csv)
- [Image-level confusion matrix](confusion_matrix.png) and [confidence distribution](confidence_distribution.png)
- [Teacher versus classifier](teacher_vs_classifier.png) and [scoring comparison](score_method_comparison.png)
- [Negative crop audit](other_gesture_audit.png) and [audit CSV](other_gesture_audit.csv)
- [Internal classifier decision matrix](classifier_confusion_matrix_normalized.png): this classifier has a negative class internally; the app does not return that class.
- [Training curves](classifier_train_val_curves.png) for the deployed classifier.

A manually reviewed classifier experiment improved standalone accuracy but did not reach 90%; see the [experiment report](CLASSIFIER_EXPERIMENT.md). Regenerate the graphs using `python training/make_training_graphs.py`, `python training/make_graphs.py`, `python training/audit_other_gestures.py`, and `python training/compare_classifier_experiment.py`.
