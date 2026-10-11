import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# ==========================================
# 1. 时区与初始化设置（默认使用北京时间 Asia/Shanghai）
# ==========================================
TIMEZONE = ZoneInfo(os.getenv("TZ", "Asia/Shanghai"))
COLLECTION_NAME = "tasks"


def init_firestore():
    """
    初始化 Google Cloud Firestore 数据库连接。
    支持两种方式（新手最友好）：
    1. 云端部署：读取环境变量 FIREBASE_CREDENTIALS_JSON（直接粘贴 JSON 内容）
    2. 本地运行：读取同目录下的 serviceAccountKey.json 文件
    """
    if not firebase_admin._apps:
        cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
        if cred_json:
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        elif os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
        else:
            # 如果在 Google Cloud Run 等环境中，可自动使用默认凭据
            firebase_admin.initialize_app()
    return firestore.client()


# ==========================================
# 2. 中文自然语言时间提取器（支持从标题自动提取截止时间）
# ==========================================
def extract_time_from_text(text: str, explicit_due_date: Optional[str] = None) -> tuple[str, str]:
    """
    从用户输入的文本中提取日期和时间，返回 (清理后的任务标题, 格式化截止时间 YYYY-MM-DD HH:MM)。
    如果 AI Agent 已经传入了 explicit_due_date，则优先使用。
    """
    now = datetime.now(TIMEZONE)

    # 如果 AI 已经帮我们提取好了标准格式时间
    if explicit_due_date and explicit_due_date.strip():
        cleaned_due = explicit_due_date.strip()
        # 如果只传了日期 YYYY-MM-DD，补全默认时间 23:59
        if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned_due):
            cleaned_due += " 23:59"
        return text.strip(), cleaned_due

    target_date = now.date()
    clean_title = text.strip()

    # 1) 提取显式日期：如 2026-10-12 或 10月12日
    ymd_match = re.search(r"(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})[日号]?", clean_title)
    md_match = re.search(r"(\d{1,2})[月/](\d{1,2})[日号]?", clean_title)

    if ymd_match:
        year, month, day = map(int, ymd_match.groups())
        target_date = datetime(year, month, day).date()
        clean_title = clean_title.replace(ymd_match.group(0), "", 1)
    elif md_match:
        month, day = map(int, md_match.groups())
        target_date = datetime(now.year, month, day).date()
        clean_title = clean_title.replace(md_match.group(0), "", 1)
    else:
        # 2) 提取相对日期词：今天、明天、后天、大后天
        relative_days = {
            "大后天": 3,
            "后天": 2,
            "明天": 1,
            "明日": 1,
            "今天": 0,
            "今日": 0,
            "今晚": 0,
        }
        for word, days in relative_days.items():
            if word in clean_title:
                target_date = (now + timedelta(days=days)).date()
                if word != "今晚":
                    clean_title = clean_title.replace(word, "", 1)
                break

    # 3) 提取具体时刻：如 "下午3点半"、"晚上8点"、"15:30"
    hour, minute = 23, 59  # 默认当天 23:59 截止
    time_pattern = re.search(
        r"(早上|上午|中午|下午|晚上|今晚|凌晨)?\s*(\d{1,2})(?:[:点时](\d{1,2}|半)?)?\s*分?",
        clean_title,
    )
    if time_pattern and (
        time_pattern.group(1)
        or ":" in time_pattern.group(0)
        or "点" in time_pattern.group(0)
        or "时" in time_pattern.group(0)
    ):
        period = time_pattern.group(1) or ""
        parsed_hour = int(time_pattern.group(2))
        parsed_min_str = time_pattern.group(3)

        if parsed_min_str == "半":
            parsed_min = 30
        elif parsed_min_str:
            parsed_min = int(parsed_min_str)
        else:
            parsed_min = 0

        if period in ("下午", "晚上", "今晚") and parsed_hour < 12:
            parsed_hour += 12
        elif period == "中午" and parsed_hour < 11:
            parsed_hour += 12

        if 0 <= parsed_hour <= 23 and 0 <= parsed_min <= 59:
            hour, minute = parsed_hour, parsed_min
            clean_title = clean_title.replace(time_pattern.group(0), "", 1)

    # 如果还有残留的“今晚”，清理掉
    if "今晚" in clean_title:
        clean_title = clean_title.replace("今晚", "", 1)

    # 清理多余连接词和空格，如果清理后为空则保留原标题
    clean_title = re.sub(r"^[的去把在，,。\s]+", "", clean_title).strip()
    if not clean_title:
        clean_title = text.strip()

    due_date_str = f"{target_date.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}"
    return clean_title, due_date_str


# ==========================================
# 3. 核心业务逻辑（供 REST API 和 MCP 共同调用）
# ==========================================
def core_add_task(title: str, due_date: Optional[str] = None) -> dict:
    """添加待办事项（自动提取时间并存入 Firestore）"""
    db = init_firestore()
    final_title, final_due_date = extract_time_from_text(title, due_date)
    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    task_data = {
        "title": final_title,
        "raw_input": title,
        "due_date": final_due_date,
        "due_day": final_due_date[:10],  # YYYY-MM-DD，方便按天查询
        "completed": False,
        "created_at": now_str,
    }

    _, doc_ref = db.collection(COLLECTION_NAME).add(task_data)
    task_data["id"] = doc_ref.id
    return {
        "success": True,
        "message": f"已添加待办：「{final_title}」，截止时间：{final_due_date}",
        "task": task_data,
    }


def core_get_today_tasks(include_completed: bool = True) -> dict:
    """获取今天截止的所有待办事项"""
    db = init_firestore()
    today_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d")

    docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("due_day", "==", today_str))
        .stream()
    )

    tasks = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        if not include_completed and data.get("completed", False):
            continue
        tasks.append(data)

    # 按截止时间排序
    tasks.sort(key=lambda x: x.get("due_date", ""))

    return {
        "success": True,
        "date": today_str,
        "count": len(tasks),
        "tasks": tasks,
    }


def core_complete_task(task_id: Optional[str] = None, title_keyword: Optional[str] = None) -> dict:
    """
    完成待办事项。
    支持通过 task_id 精确完成，也支持通过 title_keyword（标题关键词）模糊匹配完成！
    """
    db = init_firestore()
    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    # 方式1：如果提供了 task_id
    if task_id:
        doc_ref = db.collection(COLLECTION_NAME).document(task_id)
        doc = doc_ref.get()
        if not doc.exists:
            raise ValueError(f"找不到 ID 为 {task_id} 的待办任务")
        doc_ref.update({"completed": True, "completed_at": now_str})
        updated = doc.to_dict()
        updated["id"] = doc.id
        updated["completed"] = True
        return {
            "success": True,
            "message": f"已完成待办：「{updated.get('title')}」",
            "task": updated,
        }

    # 方式2：如果 AI Agent 只提供了任务标题关键词
    if title_keyword:
        docs = (
            db.collection(COLLECTION_NAME)
            .where(filter=firestore.FieldFilter("completed", "==", False))
            .stream()
        )
        for doc in docs:
            data = doc.to_dict()
            if title_keyword in data.get("title", "") or title_keyword in data.get("raw_input", ""):
                doc.reference.update({"completed": True, "completed_at": now_str})
                data["id"] = doc.id
                data["completed"] = True
                return {
                    "success": True,
                    "message": f"已完成待办：「{data.get('title')}」",
                    "task": data,
                }
        raise ValueError(f"没有找到包含「{title_keyword}」的未完成待办")

    raise ValueError("请提供 task_id（任务ID）或 title_keyword（任务标题关键词）")


# ==========================================
# 4. 创建 MCP 服务器（支持 Cursor / Claude / Cherry Studio 等 MCP 客户端）
# ==========================================
mcp = FastMCP("TodoList-MCP")


@mcp.tool()
def add_task(title: str, due_date: Optional[str] = None) -> dict:
    """
    添加一个新的待办任务，支持从文本中自动提取时间。
    :param title: 任务标题或包含时间的自然语言描述（例如："明天下午3点开周会"）
    :param due_date: 可选，格式为 YYYY-MM-DD HH:MM 或 YYYY-MM-DD。若不传则自动从 title 提取时间，默认今天 23:59。
    """
    return core_add_task(title=title, due_date=due_date)


@mcp.tool()
def get_today_tasks(include_completed: bool = True) -> dict:
    """
    获取今天截止的所有待办事项列表。
    :param include_completed: 是否包含已完成的任务，默认 True
    """
    return core_get_today_tasks(include_completed=include_completed)


@mcp.tool()
def complete_task(task_id: Optional[str] = None, title_keyword: Optional[str] = None) -> dict:
    """
    将某个待办事项标记为已完成。支持传入 task_id 或任务标题关键词 title_keyword。
    :param task_id: 待办任务的 ID（可选）
    :param title_keyword: 待办任务的标题关键词，例如 "周会"（可选）
    """
    return core_complete_task(task_id=task_id, title_keyword=title_keyword)


# ==========================================
# 5. 创建 FastAPI REST API 服务（支持 Coze / Dify / GPTs / 任何 Agent 调用）
# ==========================================
app = FastAPI(
    title="AI Agent 专属待办事项系统 (Firestore + MCP/REST)",
    description="提供 add_task、get_today_tasks、complete_task 三个标准工具接口，支持 REST API 与 MCP 协议。",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AddTaskRequest(BaseModel):
    title: str = Field(..., description="任务标题或带时间的句子，如：'今天下午4点买牛奶'")
    due_date: Optional[str] = Field(
        None, description="可选，截止时间（YYYY-MM-DD HH:MM）。不填则自动从 title 中提取时间"
    )


class CompleteTaskRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="任务 ID（与 title_keyword 二选一即可）")
    title_keyword: Optional[str] = Field(None, description="任务标题关键词，如：'买牛奶'")


@app.get("/", summary="健康检查")
def root():
    return {
        "status": "running",
        "service": "AI Agent TodoList API & MCP Server",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "mcp_sse_url": "/sse",
    }


@app.post("/add_task", operation_id="add_task", summary="添加待办（支持自动提取时间）")
def api_add_task(req: AddTaskRequest):
    try:
        return core_add_task(title=req.title, due_date=req.due_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_today_tasks", operation_id="get_today_tasks", summary="获取今天的待办")
def api_get_today_tasks(
    include_completed: bool = Query(True, description="是否包含已完成的待办任务")
):
    try:
        return core_get_today_tasks(include_completed=include_completed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/complete_task", operation_id="complete_task", summary="完成待办")
def api_complete_task(req: CompleteTaskRequest):
    try:
        return core_complete_task(task_id=req.task_id, title_keyword=req.title_keyword)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 将 MCP SSE 服务挂载到同一个 FastAPI 应用下，路径为 /sse
app.mount("/", mcp.sse_app())


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
