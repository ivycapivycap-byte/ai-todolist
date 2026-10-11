import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google import genai
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# ==========================================
# 1. 时区与初始化设置（默认使用东京时间 Asia/Tokyo）
# ==========================================
TIMEZONE = ZoneInfo(os.getenv("TZ", "Asia/Tokyo"))
COLLECTION_NAME = "tasks"
_db_client = None


def init_firestore():
    """初始化并复用全局 Google Cloud Firestore 数据库连接（避免每次请求重复握手）"""
    global _db_client
    if _db_client is not None:
        return _db_client

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
            firebase_admin.initialize_app()

    _db_client = firestore.client()
    return _db_client


# ==========================================
# 2. 中文自然语言时间提取器（支持从标题自动提取截止时间）
# ==========================================
def extract_time_from_text(text: str, explicit_due_date: Optional[str] = None) -> tuple[str, str]:
    now = datetime.now(TIMEZONE)

    if explicit_due_date and explicit_due_date.strip():
        cleaned_due = explicit_due_date.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned_due):
            cleaned_due += " 23:59"
        return text.strip(), cleaned_due

    target_date = now.date()
    clean_title = text.strip()

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

    hour, minute = 23, 59
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

    if "今晚" in clean_title:
        clean_title = clean_title.replace("今晚", "", 1)

    clean_title = re.sub(r"^[的去把在，,。\s]+", "", clean_title).strip()
    if not clean_title:
        clean_title = text.strip()

    due_date_str = f"{target_date.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}"
    return clean_title, due_date_str


# ==========================================
# 3. 核心业务逻辑（支持 WriteBatch 极速批量写入）
# ==========================================
def core_add_task(title: str, due_date: Optional[str] = None) -> dict:
    db = init_firestore()
    final_title, final_due_date = extract_time_from_text(title, due_date)
    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    doc_ref = db.collection(COLLECTION_NAME).document()
    task_data = {
        "title": final_title,
        "raw_input": title,
        "due_date": final_due_date,
        "due_day": final_due_date[:10],
        "completed": False,
        "created_at": now_str,
    }
    doc_ref.set(task_data)
    task_data["id"] = doc_ref.id
    return {
        "success": True,
        "message": f"已添加待办：「{final_title}」，截止时间：{final_due_date}",
        "task": task_data,
    }


def core_batch_add_tasks(items: list) -> dict:
    """使用 Firestore WriteBatch 一次性打包写入所有待办，速度提升 5~10 倍！"""
    db = init_firestore()
    batch = db.batch()
    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    added = []

    for item in items:
        final_title, final_due_date = extract_time_from_text(item.title, item.due_date)
        doc_ref = db.collection(COLLECTION_NAME).document()
        task_data = {
            "title": final_title,
            "raw_input": item.title,
            "due_date": final_due_date,
            "due_day": final_due_date[:10],
            "completed": False,
            "created_at": now_str,
        }
        batch.set(doc_ref, task_data)
        task_with_id = dict(task_data)
        task_with_id["id"] = doc_ref.id
        added.append(task_with_id)

    if added:
        batch.commit()

    return {"success": True, "count": len(added), "tasks": added}


def core_get_today_tasks(include_completed: bool = True) -> dict:
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

    tasks.sort(key=lambda x: (x.get("completed", False), x.get("due_date", "")))
    return {"success": True, "date": today_str, "count": len(tasks), "tasks": tasks}


def core_get_all_tasks() -> dict:
    db = init_firestore()
    now = datetime.now(TIMEZONE)
    today_str = now.strftime("%Y-%m-%d")
    tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    docs = db.collection(COLLECTION_NAME).stream()
    tasks = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        if "due_day" not in data and data.get("due_date"):
            data["due_day"] = data["due_date"][:10]
        tasks.append(data)

    tasks.sort(key=lambda x: (x.get("completed", False), x.get("due_date", "")))
    return {
        "success": True,
        "today": today_str,
        "tomorrow": tomorrow_str,
        "count": len(tasks),
        "tasks": tasks,
    }


def core_complete_task(
    task_id: Optional[str] = None,
    title_keyword: Optional[str] = None,
    completed: bool = True,
) -> dict:
    db = init_firestore()
    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    if task_id:
        doc_ref = db.collection(COLLECTION_NAME).document(task_id)
        doc_ref.update(
            {
                "completed": completed,
                "completed_at": now_str if completed else None,
            }
        )
        return {
            "success": True,
            "message": "已更新待办完成状态",
            "task_id": task_id,
            "completed": completed,
        }

    if title_keyword:
        docs = (
            db.collection(COLLECTION_NAME)
            .where(filter=firestore.FieldFilter("completed", "==", not completed))
            .stream()
        )
        for doc in docs:
            data = doc.to_dict()
            if title_keyword in data.get("title", "") or title_keyword in data.get("raw_input", ""):
                doc.reference.update(
                    {
                        "completed": completed,
                        "completed_at": now_str if completed else None,
                    }
                )
                data["id"] = doc.id
                data["completed"] = completed
                return {
                    "success": True,
                    "message": f"已完成待办：「{data.get('title')}」",
                    "task": data,
                }
        raise ValueError(f"没有找到包含「{title_keyword}」的待办任务")

    raise ValueError("请提供 task_id（任务ID）或 title_keyword（任务标题关键词）")


def core_update_task(
    task_id: str,
    title: Optional[str] = None,
    due_date: Optional[str] = None,
) -> dict:
    db = init_firestore()
    doc_ref = db.collection(COLLECTION_NAME).document(task_id)

    updates = {}
    if title is not None and title.strip():
        updates["title"] = title.strip()
    if due_date is not None and due_date.strip():
        cleaned_due = due_date.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned_due):
            cleaned_due += " 23:59"
        updates["due_date"] = cleaned_due
        updates["due_day"] = cleaned_due[:10]

    if updates:
        doc_ref.update(updates)

    return {"success": True, "message": "已更新待办任务", "task_id": task_id, "updates": updates}


def core_delete_task(
    task_id: Optional[str] = None,
    title_keyword: Optional[str] = None,
) -> dict:
    db = init_firestore()
    if task_id:
        db.collection(COLLECTION_NAME).document(task_id).delete()
        return {"success": True, "message": "已删除待办", "deleted_id": task_id}

    if title_keyword:
        docs = db.collection(COLLECTION_NAME).stream()
        for doc in docs:
            data = doc.to_dict()
            if title_keyword in data.get("title", "") or title_keyword in data.get("raw_input", ""):
                doc.reference.delete()
                return {
                    "success": True,
                    "message": f"已删除待办：「{data.get('title')}」",
                    "deleted_id": doc.id,
                }
        raise ValueError(f"没有找到包含「{title_keyword}」的待办任务")

    raise ValueError("请提供 task_id 或 title_keyword")


def core_clear_completed_tasks() -> dict:
    """使用 WriteBatch 一次性批量清空所有已完成任务"""
    db = init_firestore()
    docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("completed", "==", True))
        .stream()
    )
    batch = db.batch()
    deleted_count = 0
    for doc in docs:
        batch.delete(doc.reference)
        deleted_count += 1
    if deleted_count > 0:
        batch.commit()
    return {
        "success": True,
        "deleted_count": deleted_count,
        "message": f"已一键清空 {deleted_count} 条已完成的待办任务",
    }


# ==========================================
# 4. Gemini 多轮对话想法整理与拆解模型
# ==========================================
class SuggestedTask(BaseModel):
    title: str = Field(description="清晰、简练的具体待办事项标题（不包含日期时间词）")
    due_date: str = Field(description="建议的截止时间，格式必须为 YYYY-MM-DD HH:MM")


class ThoughtPlanResult(BaseModel):
    reply: str = Field(description="对用户的温和回复、思路梳理或针对用户修改意见的反馈说明")
    suggested_tasks: list[SuggestedTask] = Field(
        description="最新完整版的待办任务方案列表（根据对话调整后的完整清单）"
    )


class OrganizeThoughtsRequest(BaseModel):
    thoughts: str = Field(..., description="用户的本轮输入内容或修改反馈")
    previous_interaction_id: Optional[str] = Field(
        None, description="上一轮对话的 interaction_id，用于多轮连续对话"
    )
    current_draft: Optional[list[SuggestedTask]] = Field(
        None, description="当前界面上的待办草案列表"
    )
    api_key: Optional[str] = Field(None, description="可选，前端传入的 Gemini API Key")


class BatchAddTasksRequest(BaseModel):
    tasks: list[SuggestedTask]


# ==========================================
# 5. 创建 MCP 服务器
# ==========================================
mcp = FastMCP("TodoList-MCP")


@mcp.tool()
def add_task(title: str, due_date: Optional[str] = None) -> dict:
    """添加一个新的待办任务，支持从文本中自动提取时间。"""
    return core_add_task(title=title, due_date=due_date)


@mcp.tool()
def get_today_tasks(include_completed: bool = True) -> dict:
    """获取今天截止的所有待办事项列表。"""
    return core_get_today_tasks(include_completed=include_completed)


@mcp.tool()
def complete_task(task_id: Optional[str] = None, title_keyword: Optional[str] = None) -> dict:
    """将某个待办事项标记为已完成（支持传 task_id 或标题关键词 title_keyword）。"""
    return core_complete_task(task_id=task_id, title_keyword=title_keyword, completed=True)


@mcp.tool()
def delete_task(task_id: Optional[str] = None, title_keyword: Optional[str] = None) -> dict:
    """删除某个待办事项（支持传 task_id 或标题关键词 title_keyword）。"""
    return core_delete_task(task_id=task_id, title_keyword=title_keyword)


@mcp.tool()
def update_task(
    task_id: str, title: Optional[str] = None, due_date: Optional[str] = None
) -> dict:
    """修改某个待办事项的标题或截止时间（YYYY-MM-DD HH:MM）。"""
    return core_update_task(task_id=task_id, title=title, due_date=due_date)


# ==========================================
# 6. 创建 FastAPI REST API + 极速响应网页界面（v2.1 加速版）
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时提前预热 Firestore 连接，避免第一次点击时等待握手
    try:
        init_firestore()
    except Exception:
        pass
    yield


app = FastAPI(
    title="AI Agent 专属待办事项系统 (Firestore + Gemini 多轮对话 + MCP)",
    description="提供 add_task、get_today_tasks、complete_task、update_task、delete_task 等标准接口及可视化看板。",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AddTaskRequest(BaseModel):
    title: str = Field(..., description="任务标题或带时间的句子，如：'今天下午4点买牛奶'")
    due_date: Optional[str] = Field(None, description="可选，截止时间（YYYY-MM-DD HH:MM）")


class CompleteTaskRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="任务 ID（与 title_keyword 二选一）")
    title_keyword: Optional[str] = Field(None, description="任务标题关键词，如：'买牛奶'")
    completed: bool = Field(True, description="是否完成（传 false 可恢复为未完成）")


class UpdateTaskRequest(BaseModel):
    task_id: str = Field(..., description="任务 ID")
    title: Optional[str] = Field(None, description="新的任务标题")
    due_date: Optional[str] = Field(None, description="新的截止时间 YYYY-MM-DD HH:MM")


class DeleteTaskRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="任务 ID")
    title_keyword: Optional[str] = Field(None, description="任务标题关键词")


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>我的 AI 专属待办清单</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen text-slate-800 pb-10">
  <div class="max-w-xl mx-auto px-3.5 py-5 space-y-5">

    <!-- 模块一：Gemini 想法整理室（支持多轮连续对话 + 实时修改草案） -->
    <div class="bg-gradient-to-br from-indigo-600 to-violet-700 rounded-2xl shadow-md p-4 sm:p-5 text-white">
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <span class="text-xl">💬</span>
          <h2 class="text-base sm:text-lg font-bold">Gemini 想法整理室</h2>
        </div>
        <button onclick="resetChat()" class="text-xs bg-white/20 hover:bg-white/30 px-2.5 py-1 rounded-lg transition">
          🧹 新对话
        </button>
      </div>

      <!-- 微信式多轮聊天记录窗口（有对话时自动展开） -->
      <div id="chat-history" class="hidden bg-white/95 text-slate-800 rounded-xl p-3 max-h-60 overflow-y-auto space-y-2.5 mb-3 shadow-inner"></div>

      <!-- 多轮对话输入框 -->
      <form onsubmit="handleChatSubmit(event)" class="flex gap-2">
        <input
          id="thoughts-input"
          type="text"
          placeholder="输入你的想法，或告诉 Gemini 怎么调整..."
          class="flex-1 px-3.5 py-2.5 rounded-xl text-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-amber-300"
          required
        />
        <button
          type="submit"
          id="organize-btn"
          class="px-4 py-2.5 rounded-xl bg-amber-400 hover:bg-amber-300 text-slate-900 font-semibold text-sm transition whitespace-nowrap shadow-sm"
        >
          发送 💬
        </button>
      </form>

      <!-- 实时更新的待办草案卡片 -->
      <div id="plan-preview" class="hidden mt-4 bg-white text-slate-800 rounded-xl p-3.5 shadow-inner">
        <div class="flex items-center justify-between mb-2.5">
          <span class="text-xs font-bold text-indigo-700">📝 待办方案草案（可随对话修改，也可直接改字/删减）：</span>
          <button
            type="button"
            onclick="discardDraft()"
            class="text-xs px-2.5 py-1 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-600 font-medium transition shrink-0"
          >
            🗑️ 一键放弃此方案
          </button>
        </div>
        <div id="suggested-list" class="space-y-2 mb-3"></div>
        <div class="flex gap-2">
          <button
            type="button"
            onclick="discardDraft()"
            class="px-3.5 py-2.5 rounded-xl bg-slate-100 hover:bg-rose-50 text-slate-600 hover:text-rose-600 font-medium text-xs transition whitespace-nowrap"
          >
            🗑️ 全部删除
          </button>
          <button
            type="button"
            onclick="confirmBatchAdd()"
            id="batch-add-btn"
            class="flex-1 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-sm transition shadow-sm"
          >
            🚀 方案满意，一键加入待办清单！
          </button>
        </div>
      </div>
    </div>

    <!-- 模块二：我的待办清单看板（四页签 + 极速响应） -->
    <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 sm:p-5">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h1 class="text-lg sm:text-xl font-bold text-slate-900">📋 我的待办清单</h1>
          <p id="sync-status" class="text-xs text-slate-400 mt-0.5">Google Cloud Firestore 实时同步</p>
        </div>
        <button onclick="loadTasks()" class="text-xs px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition">
          🔄 刷新
        </button>
      </div>

      <!-- 单条快速输入框 -->
      <form onsubmit="handleAdd(event)" class="flex gap-2 mb-4">
        <input
          id="task-input"
          type="text"
          placeholder="快速记一条，如：明天下午3点开会..."
          class="flex-1 px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
          required
        />
        <button
          type="submit"
          id="add-btn"
          class="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-sm transition whitespace-nowrap"
        >
          + 添加
        </button>
      </form>

      <!-- 四页签切换栏（今日 / 明日 / 未来 / 已完成） -->
      <div class="grid grid-cols-4 gap-1.5 mb-4 bg-slate-100 p-1 rounded-xl">
        <button id="tab-today" onclick="switchTab('today')" class="text-xs font-semibold py-2 rounded-lg transition bg-white text-indigo-600 shadow-sm">
          📅 今日 <span id="count-today" class="ml-0.5 text-[11px] opacity-80">0</span>
        </button>
        <button id="tab-tomorrow" onclick="switchTab('tomorrow')" class="text-xs font-medium py-2 rounded-lg transition text-slate-600 hover:text-slate-900">
          🌤️ 明日 <span id="count-tomorrow" class="ml-0.5 text-[11px] opacity-75">0</span>
        </button>
        <button id="tab-future" onclick="switchTab('future')" class="text-xs font-medium py-2 rounded-lg transition text-slate-600 hover:text-slate-900">
          🚀 未来 <span id="count-future" class="ml-0.5 text-[11px] opacity-75">0</span>
        </button>
        <button id="tab-completed" onclick="switchTab('completed')" class="text-xs font-medium py-2 rounded-lg transition text-slate-600 hover:text-slate-900">
          ✅ 已完 <span id="count-completed" class="ml-0.5 text-[11px] opacity-75">0</span>
        </button>
      </div>

      <!-- 已完成页签专属：一键清空栏 -->
      <div id="completed-toolbar" class="hidden flex items-center justify-between bg-rose-50/70 border border-rose-100 rounded-xl px-3.5 py-2 mb-3">
        <span class="text-xs text-slate-500">点击左侧勾 ☑️ 可将任务恢复为未完成</span>
        <button onclick="clearCompleted()" class="text-xs font-medium text-rose-600 hover:text-rose-700 bg-white px-2.5 py-1 rounded-lg border border-rose-200 shadow-sm transition">
          🧹 一键清空已完成
        </button>
      </div>

      <!-- 任务列表容器 -->
      <div id="task-list" class="space-y-2.5">
        <p class="text-center text-slate-400 py-8 text-sm">正在加载待办事项...</p>
      </div>
    </div>
  </div>

  <!-- 编辑任务弹窗 -->
  <div id="edit-modal" class="hidden fixed inset-0 bg-black/40 flex items-center justify-center z-50 px-4">
    <div class="bg-white rounded-2xl shadow-xl max-w-sm w-full p-5 space-y-4">
      <h3 class="text-base font-bold text-slate-800">✏️ 修改待办事项</h3>
      <input type="hidden" id="edit-id" />
      <div>
        <label class="block text-xs text-slate-500 mb-1">任务标题</label>
        <input id="edit-title" type="text" class="w-full px-3 py-2 rounded-xl border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500" />
      </div>
      <div>
        <label class="block text-xs text-slate-500 mb-1">截止时间 (格式：YYYY-MM-DD HH:MM)</label>
        <input id="edit-due" type="text" class="w-full px-3 py-2 rounded-xl border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500" />
      </div>
      <div class="flex justify-end gap-2 pt-1">
        <button onclick="closeEditModal()" class="px-4 py-2 rounded-xl bg-slate-100 text-slate-600 text-xs font-medium hover:bg-slate-200">取消</button>
        <button onclick="submitEditTask()" class="px-4 py-2 rounded-xl bg-indigo-600 text-white text-xs font-medium hover:bg-indigo-700">保存修改</button>
      </div>
    </div>
  </div>

  <script>
    let currentTab = 'today';
    let allTasksCache = [];
    let serverToday = '';
    let serverTomorrow = '';
    let currentSuggestedTasks = [];
    let currentInteractionId = null;

    function setSyncStatus(text) {
      const el = document.getElementById('sync-status');
      if (el) el.innerText = text;
    }

    function sortTasksCache() {
      allTasksCache.sort((a, b) => {
        if (Boolean(a.completed) !== Boolean(b.completed)) {
          return a.completed ? 1 : -1;
        }
        return (a.due_date || '').localeCompare(b.due_date || '');
      });
    }

    function escapeHtml(str) {
      return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function appendChatBubble(role, text) {
      const chatEl = document.getElementById('chat-history');
      chatEl.classList.remove('hidden');
      const div = document.createElement('div');
      if (role === 'user') {
        div.className = 'flex items-start justify-end gap-2';
        div.innerHTML = `
          <div class="bg-indigo-600 text-white rounded-2xl rounded-tr-none px-3 py-2 text-xs leading-relaxed max-w-[82%]">
            ${escapeHtml(text)}
          </div>
          <div class="w-7 h-7 rounded-full bg-indigo-200 text-indigo-800 flex items-center justify-center text-xs font-bold shrink-0">我</div>
        `;
      } else {
        div.className = 'flex items-start gap-2';
        div.innerHTML = `
          <div class="w-7 h-7 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs font-bold shrink-0">AI</div>
          <div class="bg-slate-100 text-slate-700 rounded-2xl rounded-tl-none px-3 py-2 text-xs leading-relaxed max-w-[82%]">
            ${escapeHtml(text)}
          </div>
        `;
      }
      chatEl.appendChild(div);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function getCurrentDraftFromInputs() {
      const draft = [];
      const rows = document.querySelectorAll('.draft-item-row');
      rows.forEach(row => {
        const checkEl = row.querySelector('.draft-check');
        const titleEl = row.querySelector('.draft-title');
        const dateEl = row.querySelector('.draft-date');
        if (checkEl && checkEl.checked && titleEl && dateEl && titleEl.value.trim()) {
          draft.push({
            title: titleEl.value.trim(),
            due_date: dateEl.value.trim()
          });
        }
      });
      return draft;
    }

    function renderDraftList(tasks) {
      currentSuggestedTasks = tasks || [];
      const previewEl = document.getElementById('plan-preview');
      const listEl = document.getElementById('suggested-list');

      if (!currentSuggestedTasks || currentSuggestedTasks.length === 0) {
        previewEl.classList.add('hidden');
        listEl.innerHTML = '';
        return;
      }

      previewEl.classList.remove('hidden');
      listEl.innerHTML = currentSuggestedTasks.map((item, idx) => `
        <div class="draft-item-row flex items-center gap-1.5 p-2 rounded-lg border border-slate-200 bg-slate-50">
          <input type="checkbox" class="draft-check w-4 h-4 text-indigo-600 rounded shrink-0" checked />
          <input type="text" value="${escapeHtml(item.title)}" class="draft-title flex-1 min-w-0 bg-white px-2 py-1 rounded border border-slate-200 text-xs font-medium" />
          <input type="text" value="${escapeHtml(item.due_date)}" class="draft-date w-32 bg-white px-2 py-1 rounded border border-slate-200 text-xs text-slate-600 shrink-0" />
          <button type="button" onclick="removeDraftItem(${idx})" title="移除此条" class="p-1 text-slate-400 hover:text-rose-600 text-xs shrink-0">✕</button>
        </div>
      `).join('');
    }

    function removeDraftItem(idx) {
      const current = getCurrentDraftFromInputs();
      current.splice(idx, 1);
      renderDraftList(current);
    }

    function discardDraft() {
      currentSuggestedTasks = [];
      document.getElementById('plan-preview').classList.add('hidden');
      document.getElementById('suggested-list').innerHTML = '';
    }

    function resetChat() {
      currentInteractionId = null;
      discardDraft();
      const chatEl = document.getElementById('chat-history');
      chatEl.innerHTML = '';
      chatEl.classList.add('hidden');
    }

    async function handleChatSubmit(e) {
      e.preventDefault();
      const inputEl = document.getElementById('thoughts-input');
      const thoughts = inputEl.value.trim();
      if (!thoughts) return;

      appendChatBubble('user', thoughts);
      inputEl.value = '';

      const btn = document.getElementById('organize-btn');
      const currentDraft = getCurrentDraftFromInputs();

      btn.disabled = true;
      btn.innerText = '思考中...';

      try {
        const res = await fetch('/organize_thoughts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            thoughts,
            previous_interaction_id: currentInteractionId,
            current_draft: currentDraft
          })
        });
        const data = await res.json();
        if (!res.ok) {
          appendChatBubble('ai', '⚠️ ' + (data.detail || '整理失败，请检查 Render 环境变量中的 GEMINI_API_KEY'));
          return;
        }

        currentInteractionId = data.interaction_id || currentInteractionId;
        appendChatBubble('ai', data.reply);
        renderDraftList(data.suggested_tasks || []);
      } catch (err) {
        appendChatBubble('ai', '⚠️ 网络请求失败，请稍后重试');
      } finally {
        btn.disabled = false;
        btn.innerText = '发送 💬';
      }
    }

    async function confirmBatchAdd() {
      const selected = getCurrentDraftFromInputs();
      if (selected.length === 0) {
        alert('请至少勾选一项待办！');
        return;
      }

      // 乐观更新：0.01秒立即显示在页面上，无需等待网络来回！
      const tempTasks = selected.map((item, i) => ({
        id: 'temp-' + Date.now() + '-' + i,
        title: item.title,
        due_date: item.due_date,
        due_day: (item.due_date || '').slice(0, 10) || serverToday,
        completed: false
      }));
      allTasksCache.push(...tempTasks);
      sortTasksCache();
      renderTasks();

      document.getElementById('plan-preview').classList.add('hidden');
      currentSuggestedTasks = [];
      currentInteractionId = null;
      appendChatBubble('ai', `⚡ 已将 ${selected.length} 项待办加入下方列表（后台正在同步云端...）`);
      setSyncStatus('☁️ 正在批量同步到云端...');

      try {
        const res = await fetch('/batch_add_tasks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tasks: selected })
        });
        const data = await res.json();
        if (data.tasks) {
          // 替换临时 ID 为真实数据库 ID
          const tempIds = new Set(tempTasks.map(t => t.id));
          allTasksCache = allTasksCache.filter(t => !tempIds.has(t.id));
          allTasksCache.push(...data.tasks);
          sortTasksCache();
          renderTasks();
        }
        setSyncStatus('✅ 云端已同步');
      } catch (e) {
        setSyncStatus('⚠️ 同步异常，请点刷新');
        await loadTasks();
      }
    }

    function switchTab(tab) {
      currentTab = tab;
      ['today', 'tomorrow', 'future', 'completed'].forEach(t => {
        const btn = document.getElementById(`tab-${t}`);
        if (t === tab) {
          btn.className = 'text-xs font-semibold py-2 rounded-lg transition bg-white text-indigo-600 shadow-sm';
        } else {
          btn.className = 'text-xs font-medium py-2 rounded-lg transition text-slate-600 hover:text-slate-900';
        }
      });
      renderTasks();
    }

    function formatWeekday(dateStr) {
      const parts = (dateStr || '').split('-');
      if (parts.length !== 3) return dateStr;
      const d = new Date(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2]));
      const days = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];
      return `${Number(parts[1])}月${Number(parts[2])}日 (${days[d.getDay()]})`;
    }

    function renderTaskCard(t, isOverdue = false) {
      const timePart = (t.due_date || '').slice(11, 16) || '23:59';
      let badgeHtml = '';
      if (isOverdue) {
        badgeHtml = `<span class="text-[11px] font-medium px-2 py-0.5 rounded-md bg-rose-50 text-rose-600 border border-rose-200">⚠️ 已逾期 · ${(t.due_date || '').slice(5)}</span>`;
      } else if (t.due_day === serverToday) {
        badgeHtml = `<span class="text-xs text-slate-400">⏰ 今天 ${timePart}</span>`;
      } else if (t.due_day === serverTomorrow) {
        badgeHtml = `<span class="text-xs text-slate-400">⏰ 明天 ${timePart}</span>`;
      } else {
        badgeHtml = `<span class="text-xs text-slate-400">⏰ ${escapeHtml(t.due_date || '未设置')}</span>`;
      }

      const postponeBtn = !t.completed && (currentTab === 'today')
        ? `<button onclick="postponeTask('${t.id}')" title="顺延到明天" class="text-[11px] px-2 py-1 rounded-lg bg-amber-50 hover:bg-amber-100 text-amber-700 font-medium transition">⏩ 明天</button>`
        : '';

      return `
        <div class="flex items-center justify-between p-3 rounded-xl border ${t.completed ? 'bg-slate-50 border-slate-100 opacity-70' : isOverdue ? 'bg-rose-50/20 border-rose-200' : 'bg-white border-slate-200 hover:border-indigo-200'} transition">
          <div class="flex items-center gap-3 min-w-0 flex-1 pr-2">
            <input
              type="checkbox"
              ${t.completed ? 'checked' : ''}
              onchange="toggleTaskStatus('${t.id}', ${!t.completed})"
              class="w-5 h-5 rounded text-indigo-600 focus:ring-indigo-500 cursor-pointer shrink-0"
            />
            <div class="min-w-0 flex-1">
              <div class="text-sm font-medium truncate ${t.completed ? 'line-through text-slate-400' : 'text-slate-800'}">${escapeHtml(t.title)}</div>
              <div class="mt-0.5 flex items-center gap-2 flex-wrap">${badgeHtml}</div>
            </div>
          </div>
          <div class="flex items-center gap-1 shrink-0">
            ${postponeBtn}
            <button onclick="openEditModal('${t.id}')" title="编辑" class="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 text-xs transition">✏️</button>
            <button onclick="deleteTask('${t.id}')" title="删除" class="p-1.5 rounded-lg hover:bg-rose-50 text-slate-400 hover:text-rose-600 text-xs transition">🗑️</button>
          </div>
        </div>
      `;
    }

    function renderTasks() {
      const listEl = document.getElementById('task-list');
      const toolbarEl = document.getElementById('completed-toolbar');

      const todayTasks = allTasksCache.filter(t => !t.completed && t.due_day <= serverToday);
      const tomorrowTasks = allTasksCache.filter(t => !t.completed && t.due_day === serverTomorrow);
      const futureTasks = allTasksCache.filter(t => !t.completed && t.due_day > serverTomorrow);
      const completedTasks = allTasksCache.filter(t => t.completed);

      document.getElementById('count-today').innerText = `(${todayTasks.length})`;
      document.getElementById('count-tomorrow').innerText = `(${tomorrowTasks.length})`;
      document.getElementById('count-future').innerText = `(${futureTasks.length})`;
      document.getElementById('count-completed').innerText = `(${completedTasks.length})`;

      if (currentTab === 'completed' && completedTasks.length > 0) {
        toolbarEl.classList.remove('hidden');
      } else {
        toolbarEl.classList.add('hidden');
      }

      if (currentTab === 'today') {
        if (todayTasks.length === 0) {
          listEl.innerHTML = '<p class="text-center text-slate-400 py-8 text-sm">🎉 今天的待办都清空啦，真棒！</p>';
          return;
        }
        listEl.innerHTML = todayTasks.map(t => renderTaskCard(t, t.due_day < serverToday)).join('');
      } else if (currentTab === 'tomorrow') {
        if (tomorrowTasks.length === 0) {
          listEl.innerHTML = '<p class="text-center text-slate-400 py-8 text-sm">🌤️ 明天暂时还没有安排任务~</p>';
          return;
        }
        listEl.innerHTML = tomorrowTasks.map(t => renderTaskCard(t, false)).join('');
      } else if (currentTab === 'future') {
        if (futureTasks.length === 0) {
          listEl.innerHTML = '<p class="text-center text-slate-400 py-8 text-sm">🚀 未来还没有待办计划~</p>';
          return;
        }
        const groups = {};
        futureTasks.forEach(t => {
          const day = t.due_day || '未定日期';
          if (!groups[day]) groups[day] = [];
          groups[day].push(t);
        });
        listEl.innerHTML = Object.keys(groups).sort().map(day => `
          <div class="space-y-2 pt-1">
            <div class="text-xs font-bold text-indigo-600 bg-indigo-50/60 px-2.5 py-1 rounded-lg inline-block">
              📌 ${day} · ${formatWeekday(day)}
            </div>
            <div class="space-y-2">
              ${groups[day].map(t => renderTaskCard(t, false)).join('')}
            </div>
          </div>
        `).join('');
      } else if (currentTab === 'completed') {
        if (completedTasks.length === 0) {
          listEl.innerHTML = '<p class="text-center text-slate-400 py-8 text-sm">暂无已完成的归档任务</p>';
          return;
        }
        listEl.innerHTML = completedTasks.map(t => renderTaskCard(t, false)).join('');
      }
    }

    async function loadTasks() {
      const listEl = document.getElementById('task-list');
      try {
        const res = await fetch('/all_tasks');
        const data = await res.json();
        allTasksCache = data.tasks || [];
        serverToday = data.today || '';
        serverTomorrow = data.tomorrow || '';
        renderTasks();
      } catch (e) {
        listEl.innerHTML = '<p class="text-center text-red-500 py-8 text-sm">加载失败，请稍后刷新重试</p>';
      }
    }

    async function handleAdd(e) {
      e.preventDefault();
      const input = document.getElementById('task-input');
      const btn = document.getElementById('add-btn');
      const title = input.value.trim();
      if (!title) return;
      input.value = '';
      btn.disabled = true;
      setSyncStatus('☁️ 正在同步到云端...');
      try {
        const res = await fetch('/add_task', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title })
        });
        const data = await res.json();
        if (data.task) {
          allTasksCache.push(data.task);
          sortTasksCache();
          renderTasks();
        }
        setSyncStatus('✅ 云端已同步');
      } finally {
        btn.disabled = false;
      }
    }

    async function toggleTaskStatus(id, completed) {
      // 乐观更新：瞬间打勾/恢复，不等网络
      const task = allTasksCache.find(x => x.id === id);
      if (task) {
        task.completed = completed;
        sortTasksCache();
        renderTasks();
      }
      setSyncStatus('☁️ 正在同步状态...');
      await fetch('/complete_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: id, completed })
      });
      setSyncStatus('✅ 云端已同步');
    }

    async function postponeTask(id) {
      const task = allTasksCache.find(x => x.id === id);
      if (!task) return;
      const timePart = (task.due_date || '').slice(11, 16) || '23:59';
      const newDueDate = `${serverTomorrow} ${timePart}`;
      // 乐观更新：瞬间挪到明日
      task.due_date = newDueDate;
      task.due_day = serverTomorrow;
      sortTasksCache();
      renderTasks();
      setSyncStatus('☁️ 已顺延到明天，正在同步...');

      await fetch('/update_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: id, due_date: newDueDate })
      });
      setSyncStatus('✅ 云端已同步');
    }

    function openEditModal(id) {
      const task = allTasksCache.find(x => x.id === id);
      if (!task) return;
      document.getElementById('edit-id').value = id;
      document.getElementById('edit-title').value = task.title || '';
      document.getElementById('edit-due').value = task.due_date || '';
      document.getElementById('edit-modal').classList.remove('hidden');
    }

    function closeEditModal() {
      document.getElementById('edit-modal').classList.add('hidden');
    }

    async function submitEditTask() {
      const id = document.getElementById('edit-id').value;
      const title = document.getElementById('edit-title').value.trim();
      const due_date = document.getElementById('edit-due').value.trim();
      if (!title) return;

      const task = allTasksCache.find(x => x.id === id);
      if (task) {
        task.title = title;
        task.due_date = due_date;
        task.due_day = due_date.slice(0, 10);
        sortTasksCache();
        renderTasks();
      }
      closeEditModal();
      setSyncStatus('☁️ 正在保存修改...');

      await fetch('/update_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: id, title, due_date })
      });
      setSyncStatus('✅ 云端已同步');
    }

    async function deleteTask(id) {
      if (!confirm('确定要删除这条待办吗？')) return;
      allTasksCache = allTasksCache.filter(x => x.id !== id);
      renderTasks();
      setSyncStatus('☁️ 正在删除...');
      await fetch('/delete_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: id })
      });
      setSyncStatus('✅ 云端已同步');
    }

    async function clearCompleted() {
      if (!confirm('确定要彻底清空所有已完成的待办记录吗？')) return;
      allTasksCache = allTasksCache.filter(x => !x.completed);
      renderTasks();
      setSyncStatus('☁️ 正在批量清空...');
      await fetch('/clear_completed_tasks', { method: 'POST' });
      setSyncStatus('✅ 云端已同步');
    }

    loadTasks();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return DASHBOARD_HTML


@app.post("/organize_thoughts", include_in_schema=False)
def api_organize_thoughts(req: OrganizeThoughtsRequest):
    api_key = (req.api_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="请先在 Render 环境变量设置 GEMINI_API_KEY，或点击右上角「🔑 Key」填入密钥",
        )

    now = datetime.now(TIMEZONE)
    now_str = now.strftime("%Y-%m-%d %H:%M")
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_str = weekday_names[now.weekday()]

    draft_list = [item.model_dump() for item in (req.current_draft or [])]
    draft_json = json.dumps(draft_list, ensure_ascii=False)
    user_message = (
        f"【当前页面上的待办方案草案（必须以此为基础进行增删改）】：{draft_json}\n"
        f"【用户的最新指令/修改要求】：{req.thoughts}"
    )

    try:
        client = genai.Client(api_key=api_key)
        kwargs = {
            "model": "gemini-3.8-flash",
            "system_instruction": (
                f"你是用户的私人想法整理师与待办助理。当前东京时间 (Asia/Tokyo) 是：{now_str}（{weekday_str}）。\n"
                "你支持与用户进行多轮连续对话来打磨待办事项方案，请严格遵守以下规则：\n"
                "1. 第一轮当用户说出杂乱想法时，在 reply 中用温暖、清晰的语气帮用户梳理思路，并在 suggested_tasks 中给出第一版待办清单草案；\n"
                "2. 后续轮次当用户提出修改意见时，你必须以【当前页面上的待办方案草案】为基础进行修改：\n"
                "   - 如果用户要求**删除/去掉**某几个任务，必须从 suggested_tasks 中把它们**彻底移除**，绝不能保留！\n"
                "   - 如果用户表示对方案全部不满意、不要了、或要求**全部删除/清空**，必须将 suggested_tasks 返回为空数组 `[]`！\n"
                "   - 如果用户要求修改时间或标题，直接更新对应条目并返回剩余完整列表。\n"
                "3. 每个任务的 due_date 格式必须严格为 YYYY-MM-DD HH:MM。"
            ),
            "input": user_message,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": ThoughtPlanResult.model_json_schema(),
            },
        }
        if req.previous_interaction_id:
            kwargs["previous_interaction_id"] = req.previous_interaction_id

        interaction = client.interactions.create(**kwargs)
        result = ThoughtPlanResult.model_validate_json(interaction.output_text)
        output = result.model_dump()
        output["interaction_id"] = interaction.id
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini 整理出错：{str(e)}")


@app.post("/batch_add_tasks", include_in_schema=False)
def api_batch_add_tasks(req: BatchAddTasksRequest):
    try:
        return core_batch_add_tasks(req.tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/all_tasks", include_in_schema=False)
def api_get_all_tasks():
    try:
        return core_get_all_tasks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/complete_task", operation_id="complete_task", summary="完成或恢复待办")
def api_complete_task(req: CompleteTaskRequest):
    try:
        return core_complete_task(
            task_id=req.task_id, title_keyword=req.title_keyword, completed=req.completed
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_task", operation_id="update_task", summary="修改待办标题或截止时间")
def api_update_task(req: UpdateTaskRequest):
    try:
        return core_update_task(task_id=req.task_id, title=req.title, due_date=req.due_date)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete_task", operation_id="delete_task", summary="删除单条待办")
def api_delete_task(req: DeleteTaskRequest):
    try:
        return core_delete_task(task_id=req.task_id, title_keyword=req.title_keyword)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_completed_tasks", operation_id="clear_completed_tasks", summary="一键清空所有已完成待办")
def api_clear_completed_tasks():
    try:
        return core_clear_completed_tasks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", mcp.sse_app())


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
