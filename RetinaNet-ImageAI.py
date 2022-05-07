from imageai.Classification.Custom import ClassificationModelTrainer
model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory(r'C:\Users\vinee\Documents\CT_5')                  
model_trainer.trainModel(num_objects=2, num_experiments=2, enhance_data=True, batch_size=5, show_network_summary=True)
