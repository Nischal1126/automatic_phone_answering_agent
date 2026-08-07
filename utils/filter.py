import os
import chardet

# Input folders
input_folders = [
    r"C:\Users\nisch\OneDrive\Desktop\Crawl4ai\course_docs",
    r"C:\Users\nisch\OneDrive\Desktop\Crawl4ai\doai_output"
]

# Output folder
output_folder = r"C:\Users\nisch\OneDrive\Desktop\Crawl4ai\appended_output"
os.makedirs(output_folder, exist_ok=True)

# Only process these file types
allowed_extensions = {".md", ".txt"}

for input_folder in input_folders:

    print(f"\nProcessing folder: {input_folder}")

    for filename in os.listdir(input_folder):

        input_path = os.path.join(input_folder, filename)

        # Skip directories
        if not os.path.isfile(input_path):
            print(f"Skipping directory: {filename}")
            continue

        # Skip unsupported files
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_extensions:
            print(f"Skipping unsupported file: {filename}")
            continue

        try:
            # Detect encoding
            with open(input_path, "rb") as f:
                raw = f.read()

            result = chardet.detect(raw)
            encoding = result["encoding"] or "utf-8"

            content = raw.decode(encoding, errors="replace")

            output_path = os.path.join(output_folder, filename)

            with open(output_path, "a", encoding="utf-8") as outfile:
                outfile.write(content)
                outfile.write("\n\n")
                outfile.write("=" * 80)
                outfile.write("\n\n")

            print(f"Appended: {filename} (encoding: {encoding})")

        except Exception as e:
            print(f"Failed to process {filename}")
            print(e)