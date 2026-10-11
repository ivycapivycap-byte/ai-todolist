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
from fastapi.responses import HTMLResponse
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# ==========================================
# 1. 时区与初始化设置（默认使用北京时间 Asia/Shanghai）
# ==========================================
TIMEZONE = ZoneInfo(os.getenv("TZ", "Asia/Shanghai"))
COLLECTION_NAME = "tasks"


def init_firestore():
    """初始化 Google Cloud Firestore 数据库连接"""
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
    return firestore.client()


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
# 3. 核心业务逻辑（供 REST API、网页界面和 MCP 共同调用）
# ==========================================
def core_add_task(title: str, due_date: Optional[str] = None) -> dict:
    db = init_firestore()
    final_title, final_due_date = extract_time_from_text(title, due_date)
    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    task_data = {
        "title": final_title,
        "raw_input": title,
        "due_date": final_due_date,
        "due_day": final_due_date[:10],
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
    today_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    docs = db.collection(COLLECTION_NAME).stream()
    tasks = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        tasks.append(data)
    tasks.sort(key=lambda x: (x.get("completed", False), x.get("due_date", "")))
    return {"success": True, "today": today_str, "count": len(tasks), "tasks": tasks}


def core_complete_task(task_id: Optional[str] = None, title_keyword: Optional[str] = None) -> dict:
    db = init_firestore()
    now_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

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
# 4. 创建 MCP 服务器
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
    return core_complete_task(task_id=task_id, title_keyword=title_keyword)


# ==========================================
# 5. 创建 FastAPI REST API + 可视化网页界面
# ==========================================
app = FastAPI(
    title="AI Agent 专属待办事项系统 (Firestore + MCP/REST)",
    description="提供 add_task、get_today_tasks、complete_task 三个标准工具接口。",
    version="1.1.0",
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


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>我的 AI 专属待办清单</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen text-slate-800">
  <div class="max-w-xl mx-auto px-4 py-10">
    <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
      <div class="flex items-center justify-between mb-6">
        <div>
          <h1 class="text-2xl font-bold text-slate-900">📋 我的 AI 待办清单</h1>
          <p id="date-subtitle" class="text-sm text-slate-500 mt-1">与 Google Cloud Firestore 实时同步</p>
        </div>
        <button onclick="loadTasks()" class="text-sm px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition">
          🔄 刷新
        </button>
      </div>

      <!-- 智能输入框 -->
      <form onsubmit="handleAdd(event)" class="flex gap-2 mb-6">
        <input
          id="task-input"
          type="text"
          placeholder="输入一句话，如：今天下午4点半拿快递..."
          class="flex-1 px-4 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
          required
        />
        <button
          type="submit"
          id="add-btn"
          class="px-5 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-sm transition whitespace-nowrap"
        >
          + 添加
        </button>
      </form>

      <!-- 视图切换 -->
      <div class="flex gap-2 mb-4 border-b border-slate-100 pb-3">
        <button id="tab-today" onclick="switchTab('today')" class="text-sm font-medium px-3 py-1.5 rounded-lg bg-indigo-50 text-indigo-600">
          📅 今日待办
        </button>
        <button id="tab-all" onclick="switchTab('all')" class="text-sm font-medium px-3 py-1.5 rounded-lg text-slate-600 hover:bg-slate-100">
          🗂️ 全部待办
        </button>
      </div>

      <!-- 任务列表 -->
      <div id="task-list" class="space-y-2.5">
        <p class="text-center text-slate-400 py-8 text-sm">正在加载待办事项...</p>
      </div>
    </div>
  </div>

  <script>
    let currentTab = 'today';

    function switchTab(tab) {
      currentTab = tab;
      document.getElementById('tab-today').className = tab === 'today'
        ? 'text-sm font-medium px-3 py-1.5 rounded-lg bg-indigo-50 text-indigo-600'
        : 'text-sm font-medium px-3 py-1.5 rounded-lg text-slate-600 hover:bg-slate-100';
      document.getElementById('tab-all').className = tab === 'all'
        ? 'text-sm font-medium px-3 py-1.5 rounded-lg bg-indigo-50 text-indigo-600'
        : 'text-sm font-medium px-3 py-1.5 rounded-lg text-slate-600 hover:bg-slate-100';
      loadTasks();
    }

    async function loadTasks() {
      const listEl = document.getElementById('task-list');
      const endpoint = currentTab === 'today' ? '/get_today_tasks' : '/all_tasks';
      try {
        const res = await fetch(endpoint);
        const data = await res.json();
        const tasks = data.tasks || [];
        if (tasks.length === 0) {
          listEl.innerHTML = '<p class="text-center text-slate-400 py-8 text-sm">🎉 暂无待办事项，休息一下吧！</p>';
          return;
        }
        listEl.innerHTML = tasks.map(t => `
          <div class="flex items-center justify-between p-3.5 rounded-xl border ${t.completed ? 'bg-slate-50 border-slate-100 opacity-60' : 'bg-white border-slate-200 hover:border-indigo-200'} transition">
            <div class="flex items-center gap-3">
              <input
                type="checkbox"
                ${t.completed ? 'checked disabled' : ''}
                onchange="completeTask('${t.id}')"
                class="w-5 h-5 rounded text-indigo-600 focus:ring-indigo-500 cursor-pointer"
              />
              <div>
                <div class="text-sm font-medium ${t.completed ? 'line-through text-slate-400' : 'text-slate-800'}">${t.title}</div>
                <div class="text-xs text-slate-400 mt-0.5">⏰ 截止：${t.due_date || '未设置'}</div>
              </div>
            </div>
            <span class="text-xs px-2.5 py-1 rounded-full ${t.completed ? 'bg-emerald-50 text-emerald-600' : 'bg-amber-50 text-amber-600'}">
              ${t.completed ? '已完成' : '待办'}
            </span>
          </div>
        `).join('');
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
      btn.disabled = true;
      btn.innerText = '添加中...';
      try {
        await fetch('/add_task', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title })
        });
        input.value = '';
        await loadTasks();
      } finally {
        btn.disabled = false;
        btn.innerText = '+ 添加';
      }
    }

    async function completeTask(id) {
      await fetch('/complete_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: id })
      });
      await loadTasks();
    }

    loadTasks();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return DASHBOARD_HTML


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


@app.post("/complete_task", operation_id="complete_task", summary="完成待办")
def api_complete_task(req: CompleteTaskRequest):
    try:
        return core_complete_task(task_id=req.task_id, title_keyword=req.title_keyword)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", mcp.sse_app())


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
