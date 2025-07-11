from utils.load_model import load_model_and_threshold

# Load the model and threshold
model, threshold, similarity_fn = load_model_and_threshold()

print("🧠 Quora Duplicate Question Detector (Terminal Mode)")
print("---------------------------------------------------")

try:
    while True:
        q1 = input("\nEnter Question 1: ").strip()
        q2 = input("Enter Question 2: ").strip()

        if not q1 or not q2:
            print("⚠️ Both questions must be non-empty. Try again.")
            continue

        score = similarity_fn(q1, q2)
        print(f"\n📈 Similarity Score: {score:.2f}")

        if score >= threshold:
            print("✅ Result: Duplicate Detected!")
            print("---------------------------------------------------")
        else:
            print("❌ Result: Not Duplicate")
            print("---------------------------------------------------")

except KeyboardInterrupt:
    print("\n👋 Exiting gracefully. Goodbye!")