"""Flask web interface for CIFAKE prediction, Grad-CAM, and reports."""

from __future__ import annotations

import io
import gc
import os
import threading
from datetime import datetime

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from itsdangerous import BadSignature, URLSafeSerializer
from PIL import Image
from werkzeug.utils import secure_filename

from src import config


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_DIR = os.path.join(BASE_DIR, "static", "gradcam")
ALLOWED_EXTENSIONS = {"bmp", "gif", "jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SECRET_KEY"] = os.environ.get(
    "CIFAKE_SECRET_KEY", "cifake-local-result-token-key"
)

_model = None
_model_lock = threading.Lock()
_serializer = URLSafeSerializer(app.config["SECRET_KEY"], salt="cifake-result")


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from src.model import load_trained_model

                _model = load_trained_model(config.FINAL_MODEL_PATH)
    return _model


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}"


def _result_token(record: dict) -> str:
    return _serializer.dumps(record)


def _record_from_token(token: str | None) -> dict | None:
    if not token:
        return None
    try:
        record = _serializer.loads(token)
    except BadSignature:
        return None
    return record if isinstance(record, dict) else None


def _record_view(record: dict, token: str) -> dict:
    view = dict(record)
    view["confidence_percent"] = f"{view['confidence'] * 100:.2f}%"
    view["real_percent"] = _format_percent(view["real_probability"])
    view["fake_percent"] = _format_percent(view["fake_probability"])
    view["thumbnail_url"] = url_for("static", filename=f"uploads/{view['stored_filename']}")
    view["uploaded_image_url"] = view["thumbnail_url"]
    if view.get("gradcam_filename"):
        view["gradcam_image_url"] = url_for(
            "static", filename=f"gradcam/{view['gradcam_filename']}"
        )
    else:
        view["gradcam_image_url"] = None

    view["result_url"] = url_for("result", token=token)
    view["report_download_url"] = url_for("download_report", token=token)
    view["history_record"] = {
        "id": view["id"],
        "originalFilename": view["original_filename"],
        "uploadedImageUrl": view["uploaded_image_url"],
        "gradcamImageUrl": view["gradcam_image_url"],
        "predictionLabel": view["predicted_label"],
        "confidence": view["confidence"],
        "realProbability": view["real_probability"],
        "fakeProbability": view["fake_probability"],
        "timestamp": view["created_at"],
        "reportDownloadUrl": view["report_download_url"],
        "resultUrl": view["result_url"],
    }
    return view


@app.errorhandler(400)
def bad_request(error):
    return render_template("404.html", message=str(error)), 400


@app.errorhandler(413)
def request_too_large(error):
    return render_template("404.html", message="Image must be 10 MB or smaller."), 413


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}, 200


@app.get("/upload")
def upload():
    return render_template("upload.html", model_ready=os.path.isfile(config.FINAL_MODEL_PATH))


@app.post("/predict")
def predict():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename or not _allowed_file(uploaded.filename):
        abort(400, "Upload a BMP, GIF, JPG, JPEG, PNG, or WEBP image.")

    original_filename = secure_filename(uploaded.filename)
    stored_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{original_filename}"
    image_path = os.path.join(UPLOAD_DIR, stored_filename)
    uploaded.save(image_path)

    try:
        with Image.open(image_path) as image:
            image.verify()
        model = _get_model()
        from src.gradcam import explain_image
        from src.predict import predict_single_image

        prediction = predict_single_image(model, image_path)
        gradcam_name = (
            f"{os.path.splitext(stored_filename)[0]}_gradcam_"
            f"{prediction['predicted_label']}.png"
        )
        if os.environ.get("CIFAKE_ENABLE_GRADCAM", "1") == "1":
            explain_image(
                model,
                image_path,
                output_dir=GRADCAM_DIR,
                real_probability=prediction["real_probability"],
            )
        else:
            gradcam_name = None
        generated_gradcam = next(
            (
                name
                for name in os.listdir(GRADCAM_DIR)
                if gradcam_name
                and name.startswith(os.path.splitext(stored_filename)[0] + "_gradcam_")
            ),
            gradcam_name,
        )
    except Exception:
        if os.path.exists(image_path):
            os.remove(image_path)
        raise

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "id": os.path.splitext(stored_filename)[0],
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "predicted_label": prediction["predicted_label"],
        "real_probability": prediction["real_probability"],
        "fake_probability": prediction["fake_probability"],
        "confidence": prediction["predicted_confidence"],
        "gradcam_filename": generated_gradcam,
        "created_at": now,
    }
    gc.collect()
    return redirect(url_for("result", token=_result_token(record)))


@app.get("/result")
def result():
    token = request.args.get("token")
    record = _record_from_token(token)
    if record is None:
        abort(404)
    return render_template("result.html", **_record_view(record, token))


@app.get("/result/<int:record_id>")
def legacy_result(record_id: int):
    abort(404)


@app.get("/history")
def history():
    return render_template("history.html")


@app.get("/report")
def download_report():
    token = request.args.get("token")
    record = _record_from_token(token)
    if record is None:
        abort(404)

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as ReportImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    view = _record_view(record, token)
    pdf_buffer = io.BytesIO()
    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("CIFAKE Image Analysis Report", styles["Title"]),
        Spacer(1, 0.15 * inch),
        Paragraph(f"File: {view['original_filename']}", styles["BodyText"]),
        Paragraph(f"Created: {view['created_at']}", styles["BodyText"]),
        Spacer(1, 0.18 * inch),
        Paragraph(f"Verdict: <b>{view['predicted_label']}</b>", styles["Heading2"]),
        Paragraph(f"Confidence: {view['confidence_percent']}", styles["BodyText"]),
        Paragraph(f"REAL probability: {view['real_percent']}%", styles["BodyText"]),
        Paragraph(f"FAKE probability: {view['fake_percent']}%", styles["BodyText"]),
        Spacer(1, 0.2 * inch),
    ]

    original_path = os.path.join(UPLOAD_DIR, view["stored_filename"])
    images = [("Original image", original_path)]
    if view.get("gradcam_filename"):
        images.append(("Grad-CAM overlay", os.path.join(GRADCAM_DIR, view["gradcam_filename"])))

    for label, path in images:
        if os.path.isfile(path):
            story.append(Paragraph(label, styles["Heading3"]))
            story.append(
                ReportImage(path, width=3.2 * inch, height=2.4 * inch, kind="proportional")
            )
            story.append(Spacer(1, 0.15 * inch))

    document.build(story)
    pdf_buffer.seek(0)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"cifake_report_{view['id']}.pdf",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
