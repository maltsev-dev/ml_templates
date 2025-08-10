import scripts.download_data
import scripts.train
import scripts.evaluate
import scripts.convert_tflite

print("✅ DataSet")
scripts.download_data.download()

print("🧠 Training")
scripts.train.main()

print("📉 Evaluation")
scripts.evaluate.main()

print("🔄 TFLite Converting")
scripts.convert_tflite.convert_to_tflite()
