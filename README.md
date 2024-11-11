
# Image-Captioning-DL-Model

This repository contains a Flask web application that generates descriptive captions for user-uploaded images using a deep learning model trained from scratch. The project was developed for the Board Infinity Hackathon.

## Features

- **Image Upload**: Users can upload images through the web interface.
- **Caption Generation**: The application processes the uploaded image and generates a descriptive caption.
- **Deep Learning Model**: Utilizes a Long Short-Term Memory (LSTM) network trained from scratch for caption generation.

## Installation

To set up the application locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/aayush9400/Image-Captioning-DL-Model.git
   cd Image-Captioning-DL-Model
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   Ensure you have all necessary dependencies by installing them with pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Weights and Vocabulary Files**:

   The model requires specific weight and vocabulary files to function correctly. Ensure that the following files are present in the repository:

   - `model.h5`
   - `vocab.npy`

   If these files are not included in the repository, you may need to contact the repository owner or refer to the project's documentation for instructions on obtaining them.

## Usage

1. **Run the Flask Application**:

   Start the Flask server by executing:

   ```bash
   python app.py
   ```

2. **Access the Web Interface**:

   Open your web browser and navigate to `http://127.0.0.1:5000/` to access the application.

3. **Upload an Image**:

   Use the provided interface to upload an image.

4. **Generate Caption**:

   After uploading, the application will process the image and display the generated caption.

## Project Structure

The repository is organized as follows:

- `app.py`: Main Flask application file.
- `model.h5`: Trained LSTM model weights.
- `vocab.npy`: Vocabulary file used by the model.
- `templates/`: Directory containing HTML templates for the web interface.
- `static/`: Directory for static files (e.g., CSS, JavaScript).
- `requirements.txt`: List of Python dependencies.

## Dependencies

The application relies on the following Python libraries:

- Flask
- NumPy
- TensorFlow/Keras
- Pillow

Ensure these are installed in your environment.

## Acknowledgments

This project was developed as part of the Board Infinity Hackathon. Special thanks to the organizers and contributors who made this project possible.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

For any issues or contributions, please open an issue or submit a pull request. 
