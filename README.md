# Nail Edge AI

A real-time nail try-on application using computer vision and hand tracking.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Create a "Nails" folder and add your nail.png image.

4. Run the application:
```bash
python app.py
```

5. Access the application:
   - Open https://localhost:5000 in your browser
   - For mobile access, use your computer's local IP address (e.g., https://192.168.1.100:5000)
   - Accept the self-signed certificate warning

## Notes
- The application requires HTTPS for camera access on mobile devices
- First-time launch will generate SSL certificates automatically
- Supported browsers: Chrome, Firefox, Safari (iOS)
