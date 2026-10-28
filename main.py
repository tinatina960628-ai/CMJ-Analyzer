
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import traceback
from fastapi.responses import FileResponse
from pathlib import Path
from video_processor import (
    analyze_video,
    export_results_csv,
    export_annotated_video,
)
import shutil
from report_generator import generate_cmj_report

app = FastAPI(title="CMJ 動作分析工具")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CMJ 動作分析工具</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 760px;
                margin: 40px auto;
                padding: 20px;
                line-height: 1.7;
            }
            .panel {
                border: 1px solid #ccc;
                border-radius: 12px;
                padding: 24px;
            }
            button {
                padding: 10px 18px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <h1>CMJ 動作分析工具</h1>
        <div class="panel">
            <h2>第一步：上傳跳躍影片</h2>
            <p>請選擇 CMJ（反向跳躍）影片。</p>
            <form action="/upload" method="post"
                  enctype="multipart/form-data">
                <input type="file" name="video"
                       accept="video/*" required>
                <br><br>
                <button type="submit">上傳影片</button>
            </form>
            <p>支援常見影片格式，例如 MP4、MOV。</p>
        </div>
    </body>
    </html>
    """

@app.post("/upload", response_class=HTMLResponse)
async def upload_video(video: UploadFile = File(...)):
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    suffix = Path(video.filename or "").suffix.lower()

    if suffix not in allowed_extensions:
        return HTMLResponse(
            "<h2>不支援的影片格式</h2>"
            "<p>請選擇 MP4、MOV、AVI 或 MKV 影片。</p>"
            '<a href="/">返回</a>',
            status_code=400
        )

    safe_name = Path(video.filename).name
    destination = UPLOAD_DIR / safe_name

    with destination.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    
    try:
        
        analysis = analyze_video(destination)
        
        csv_path = UPLOAD_DIR / f"{destination.stem}_analysis.csv"
        export_results_csv(analysis, csv_path)

        pdf_path = UPLOAD_DIR / f"{destination.stem}_report.pdf"
        generate_cmj_report(csv_path, pdf_path)
        
        video_path = UPLOAD_DIR / f"{destination.stem}_annotated.mp4"
        export_annotated_video(destination, video_path)


        results = analysis["results"]
        detected_frames = int(results["pose_detected"].sum())

        return HTMLResponse(
            f"""
         <h2>影片分析完成！</h2>
         <p>檔案名稱：{safe_name}</p>
         <p>影片 FPS：{analysis['fps']:.2f}</p>
         <p>影片影格數：{analysis['frame_count']}</p>
         <p>影片長度：{analysis['duration_seconds']:.2f} 秒</p>
         <p>成功偵測人體的影格：{detected_frames}</p>

         <h3>CMJ 人體動作分析影片</h3>
         <video controls playsinline style="max-width:100%; width:800px;">
            <source src="/video/{video_path.name}" type="video/mp4">
             您的瀏覽器不支援影片播放。
         </video>
         
         <p>
             <a href="/video/{video_path.name}" download>
            下載骨架分析影片
             </a>
        </p>
         <p><a href="/download/{csv_path.name}">下載 CSV 分析結果</a></p>
         <p><a href="/download/{pdf_path.name}">下載 PDF 分析報告</a></p>
        <p><a href="/">返回首頁</a></p>
        """
     )

    except Exception as exc:
        traceback.print_exc()
        return HTMLResponse(
            f"""
            <h2>影片分析失敗</h2>
            <p>錯誤訊息：{str(exc)}</p>
            <p><a href="/">返回首頁</a></p>
            """,
            status_code=500,
        )


@app.get("/download/{filename}")
def download_file(filename: str):
    # 只允許下載分析 CSV 或 PDF 報告
    allowed_file = (
        filename.endswith("_analysis.csv")
        or filename.endswith("_report.pdf")
    )

    if not allowed_file or Path(filename).name != filename:
        return HTMLResponse(
            "<h2>無效的下載檔案</h2>",
            status_code=400,
        )


    csv_path = UPLOAD_DIR / filename

    if not csv_path.is_file():
        return HTMLResponse(
            "<h2>找不到分析結果，請先上傳影片。</h2>",
            status_code=404,
        )

    
    media_type = (
        "text/csv"
        if filename.endswith("_analysis.csv")
        else "application/pdf"
    )

    return FileResponse(
        path=csv_path,
        media_type=media_type,
        filename=filename,
    )


@app.get("/video/{filename}")
def view_annotated_video(filename: str):
    if (
        Path(filename).name != filename
        or not filename.endswith("_annotated.mp4")
    ):
        return HTMLResponse(
            "不允許存取此檔案",
            status_code=400,
        )

    video_path = UPLOAD_DIR / filename

    if not video_path.is_file():
        return HTMLResponse(
            "找不到分析影片",
            status_code=404,
        )

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=filename,
    )
