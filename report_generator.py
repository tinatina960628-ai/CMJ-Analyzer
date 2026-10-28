
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# 繁體中文字型
FONT_PATH = Path(r"C:\Windows\Fonts\msjh.ttc")


def generate_cmj_report(csv_path, pdf_path):
    """讀取 CMJ CSV，產生繁體中文 PDF 報告。"""

    csv_path = Path(csv_path)
    pdf_path = Path(pdf_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 CSV 檔案：{csv_path}")

    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"找不到中文字型：{FONT_PATH}")

    # 註冊中文字型
    if "ChineseFont" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(
            TTFont("ChineseFont", str(FONT_PATH), subfontIndex=0)
        )

    data = pd.read_csv(csv_path)

    required_columns = [
        "right_knee_angle_deg",
        "left_knee_angle_deg",
        "right_hip_angle_deg",
        "left_hip_angle_deg",
        "pose_detected",
    ]

    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要欄位：{', '.join(missing)}")

    # 將角度欄位轉成數值，忽略無效資料
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    detected = data["pose_detected"].fillna(0).astype(bool)
    detected_count = int(detected.sum())
    total_frames = len(data)
    detection_rate = (
        detected_count / total_frames * 100 if total_frames else 0
    )

    def average_angle(column):
        values = data.loc[detected, column].dropna()
        if values.empty:
            return None
        return float(values.mean())

    def format_angle(value):
        if value is None:
            return "無有效資料"
        return f"{value:.1f} 度"

    right_knee = average_angle("right_knee_angle_deg")
    left_knee = average_angle("left_knee_angle_deg")
    right_hip = average_angle("right_hip_angle_deg")
    left_hip = average_angle("left_hip_angle_deg")

    # 建立 PDF 文件
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChineseTitle",
        parent=styles["Title"],
        fontName="ChineseFont",
        fontSize=20,
        leading=28,
        alignment=1,
        textColor=colors.HexColor("#17365D"),
    )
    body_style = ParagraphStyle(
        "ChineseBody",
        parent=styles["BodyText"],
        fontName="ChineseFont",
        fontSize=11,
        leading=18,
    )
    heading_style = ParagraphStyle(
        "ChineseHeading",
        parent=styles["Heading2"],
        fontName="ChineseFont",
        fontSize=14,
        leading=20,
        textColor=colors.HexColor("#17365D"),
    )

    story = [
        Paragraph("CMJ 反向跳躍分析報告", title_style),
        Spacer(1, 0.4 * cm),
        Paragraph("一、影片與人體偵測摘要", heading_style),
        Spacer(1, 0.2 * cm),
        Paragraph(f"CSV 檔案：{csv_path.name}", body_style),
        Paragraph(f"影片總影格數：{total_frames}", body_style),
        Paragraph(f"成功偵測人體的影格：{detected_count}", body_style),
        Paragraph(f"人體偵測成功率：{detection_rate:.1f}%", body_style),
        Spacer(1, 0.4 * cm),
        Paragraph("二、關節角度摘要", heading_style),
        Spacer(1, 0.2 * cm),
    ]

    table_data = [
        ["分析項目", "平均角度"],
        ["右膝關節", format_angle(right_knee)],
        ["左膝關節", format_angle(left_knee)],
        ["右髖關節", format_angle(right_hip)],
        ["左髖關節", format_angle(left_hip)],
    ]

    table = Table(table_data, colWidths=[7 * cm, 7 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "ChineseFont"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)

    story.extend(
        [
            Spacer(1, 0.5 * cm),
            Paragraph("三、結果解讀", heading_style),
            Spacer(1, 0.2 * cm),
            Paragraph(
                "本報告的關節角度為人體成功偵測影格中的平均值。"
                "平均值代表整段影片的整體概況，不代表最低下蹲角度、"
                "起跳瞬間角度或落地角度。",
                body_style,
            ),
            Spacer(1, 0.2 * cm),
            Paragraph(
                "人體偵測成功率反映系統在多少影格中取得人體姿態，"
                "不等同於動作分析的準確率。若成功率偏低，建議檢查"
                "拍攝角度、光線、遮擋及全身是否入鏡。",
                body_style,
            ),
            Spacer(1, 0.5 * cm),
            Paragraph("四、注意事項", heading_style),
            Spacer(1, 0.2 * cm),
            Paragraph(
                "本報告屬於影片姿態估計的初步分析，"
                "不是醫療診斷或正式運動能力評估。"
                "單一視角的二維影像可能無法完整呈現真實三維關節角度。",
                body_style,
            ),
        ]
    )

    doc.build(story)
    return pdf_path
