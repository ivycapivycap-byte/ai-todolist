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
from google import genai
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
# 4. Gemini 想法整理与拆解模型
# ==========================================
class SuggestedTask(BaseModel):
    title: str = Field(description="清晰、简练的具体待办事项标题（不包含日期时间词）")
    due_date: str = Field(description="建议的截止时间，格式必须为 YYYY-MM-DD HH:MM")


class ThoughtPlanResult(BaseModel):
    reply: str = Field(description="用温暖、有条理的语气帮用户梳理想法、缓解焦虑并说明拆解思路")
    suggested_tasks: list[SuggestedTask] = Field(description="拆解出的具体待办任务清单")


class OrganizeThoughtsRequest(BaseModel):
    thoughts: str = Field(..., description="用户的凌乱想法或碎碎念")
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
    return core_complete_task(task_id=task_id, title_keyword=title_keyword)


# ==========================================
# 6. 创建 FastAPI REST API + 可视化网页界面
# ==========================================
app = FastAPI(
    title="AI Agent 专属待办事项系统 (Firestore + Gemini + MCP)",
    description="提供 add_task、get_today_tasks、complete_task 工具接口及 Gemini 想法整理室。",
    version="1.2.0",
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
  <div class="max-w-xl mx-auto px-4 py-6 space-y-5">

    <!-- 模块一：Gemini 想法整理室（微信语音/打字倾诉 -> 一键生成待办） -->
    <div class="bg-gradient-to-br from-indigo-600 to-violet-700 rounded-2xl shadow-md p-5 text-white">
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <span class="text-xl">✨</span>
          <h2 class="text-lg font-bold">Gemini 想法整理室</h2>
        </div>
        <button onclick="toggleKeyModal()" class="text-xs bg-white/20 hover:bg-white/30 px-2.5 py-1 rounded-lg transition">
          🔑 设置 Key
        </button>
      </div>
      <p class="text-xs text-indigo-100 mb-3">
        脑子很乱？把你想做的事随便说出来（在微信里可直接用语音转文字），Gemini 先帮你理清思路，再一键生成待办！
      </p>

      <!-- 隐藏的 API Key 设置框（直接存在手机本地，不用去 Render 找环境变量也行！） -->
      <div id="key-box" class="hidden bg-black/20 rounded-xl p-3 mb-3">
        <label class="block text-xs text-indigo-100 mb-1">填入你的免费 Gemini API Key（会自动保存在当前手机/浏览器）：</label>
        <div class="flex gap-2">
          <input id="gemini-key-input" type="password" placeholder="粘贴 AIzaSy... 开头的密钥" class="flex-1 px-3 py-1.5 rounded-lg text-slate-800 text-xs" />
          <button onclick="saveApiKey()" class="bg-emerald-500 hover:bg-emerald-600 px-3 py-1.5 rounded-lg text-xs font-medium">保存</button>
        </div>
      </div>

      <textarea
        id="thoughts-input"
        rows="3"
        placeholder="例如：今天事情好多，下午3点要跟客户开会，晚上想抽空去跑个步，对了家里洗发水快没了得买，还有周五前要把报销单交了..."
        class="w-full p-3 rounded-xl text-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-amber-300 resize-none"
      ></textarea>

      <button
        onclick="organizeThoughts()"
        id="organize-btn"
        class="w-full mt-2.5 py-2.5 rounded-xl bg-white text-indigo-700 font-semibold text-sm hover:bg-indigo-50 transition shadow-sm"
      >
        🧠 让 Gemini 帮我整理并拆解待办
      </button>

      <!-- Gemini 整理结果预览与一键创建区 -->
      <div id="plan-preview" class="hidden mt-4 bg-white text-slate-800 rounded-xl p-4 shadow-inner">
        <div id="gemini-reply" class="text-sm text-slate-700 bg-indigo-50/70 p-3 rounded-lg mb-3 leading-relaxed"></div>
        <div class="text-xs font-semibold text-slate-500 mb-2">👇 勾选要加入清单的任务（可修改文字或时间）：</div>
        <div id="suggested-list" class="space-y-2 mb-3"></div>
        <button
          onclick="confirmBatchAdd()"
          id="batch-add-btn"
          class="w-full py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-sm transition"
        >
          🚀 确认无误，一键加入我的待办清单！
        </button>
      </div>
    </div>

    <!-- 模块二：我的待办清单看板 -->
    <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h1 class="text-xl font-bold text-slate-900">📋 我的待办清单</h1>
          <p class="text-xs text-slate-400 mt-0.5">Google Cloud Firestore 实时同步</p>
        </div>
        <button onclick="loadTasks()" class="text-xs px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition">
          🔄 刷新
        </button>
      </div>

      <!-- 单条快速输入框 -->
      <form onsubmit="handleAdd(event)" class="flex gap-2 mb-5">
        <input
          id="task-input"
          type="text"
          placeholder="快速记一条，如：今晚8点看书..."
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
    let currentSuggestedTasks = [];

    // 初始化读取本地保存的 Gemini Key
    document.getElementById('gemini-key-input').value = localStorage.getItem('GEMINI_API_KEY') || '';

    function toggleKeyModal() {
      const box = document.getElementById('key-box');
      box.classList.toggle('hidden');
    }

    function saveApiKey() {
      const val = document.getElementById('gemini-key-input').value.trim();
      localStorage.setItem('GEMINI_API_KEY', val);
      document.getElementById('key-box').classList.add('hidden');
      alert('已保存 Gemini API Key！现在可以直接整理想法啦。');
    }

    async function organizeThoughts() {
      const thoughts = document.getElementById('thoughts-input').value.trim();
      if (!thoughts) {
        alert('请先输入你的想法或要做的事~');
        return;
      }
      const btn = document.getElementById('organize-btn');
      const localKey = localStorage.getItem('GEMINI_API_KEY') || '';
      btn.disabled = true;
      btn.innerText = '🤔 Gemini 正在帮你梳理思路...';

      try {
        const res = await fetch('/organize_thoughts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ thoughts, api_key: localKey })
        });
        const data = await res.json();
        if (!res.ok) {
          if (res.status === 400) {
            document.getElementById('key-box').classList.remove('hidden');
          }
          alert(data.detail || '整理失败，请检查 Gemini API Key');
          return;
        }

        document.getElementById('plan-preview').classList.remove('hidden');
        document.getElementById('gemini-reply').innerText = '💡 ' + data.reply;
        currentSuggestedTasks = data.suggested_tasks || [];

        const listEl = document.getElementById('suggested-list');
        listEl.innerHTML = currentSuggestedTasks.map((item, idx) => `
          <div class="flex items-center gap-2 p-2.5 rounded-lg border border-slate-200 bg-slate-50">
            <input type="checkbox" id="sug-check-${idx}" checked class="w-4 h-4 text-indigo-600 rounded" />
            <input type="text" id="sug-title-${idx}" value="${item.title}" class="flex-1 bg-white px-2 py-1 rounded border border-slate-200 text-xs font-medium" />
            <input type="text" id="sug-date-${idx}" value="${item.due_date}" class="w-32 bg-white px-2 py-1 rounded border border-slate-200 text-xs text-slate-600" />
          </div>
        `).join('');
      } catch (e) {
        alert('网络请求失败，请稍后重试');
      } finally {
        btn.disabled = false;
        btn.innerText = '🧠 让 Gemini 帮我整理并拆解待办';
      }
    }

    async function confirmBatchAdd() {
      const selected = [];
      currentSuggestedTasks.forEach((_, idx) => {
        const checked = document.getElementById(`sug-check-${idx}`).checked;
        if (checked) {
          selected.push({
            title: document.getElementById(`sug-title-${idx}`).value.trim(),
            due_date: document.getElementById(`sug-date-${idx}`).value.trim()
          });
        }
      });

      if (selected.length === 0) {
        alert('请至少勾选一项待办！');
        return;
      }

      const btn = document.getElementById('batch-add-btn');
      btn.disabled = true;
      btn.innerText = '正在一键写入云端数据库...';
      try {
        await fetch('/batch_add_tasks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tasks: selected })
        });
        document.getElementById('thoughts-input').value = '';
        document.getElementById('plan-preview').classList.add('hidden');
        await loadTasks();
      } finally {
        btn.disabled = false;
        btn.innerText = '🚀 确认无误，一键加入我的待办清单！';
      }
    }

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


@app.post("/organize_thoughts", include_in_schema=False)
def api_organize_thoughts(req: OrganizeThoughtsRequest):
    api_key = (req.api_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="请先点击右上角「🔑 设置 Key」填入你的 Gemini API Key（或在 Render 环境变量中设置 GEMINI_API_KEY）",
        )

    now = datetime.now(TIMEZONE)
    now_str = now.strftime("%Y-%m-%d %H:%M")
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_str = weekday_names[now.weekday()]

    try:
        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            system_instruction=(
                f"你是用户的私人想法整理师与待办助理。当前北京时间是：{now_str}（{weekday_str}）。\n"
                "当用户倾诉一堆杂乱的想法、计划或待办事项时：\n"
                "1. 请先在 reply 中用温暖、清晰的语气帮用户梳理思路，给出轻重缓急的建议；\n"
                "2. 将所有需要做的具体行动拆解到 suggested_tasks 列表中；\n"
                "3. 为每个任务推断合理的截止时间 due_date（格式必须为 YYYY-MM-DD HH:MM，若用户未提具体时刻，可根据事项性质安排在 10:00、15:00、20:00 或 23:59）。"
            ),
            input=req.thoughts,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": ThoughtPlanResult.model_json_schema(),
            },
        )
        result = ThoughtPlanResult.model_validate_json(interaction.output_text)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini 整理出错：{str(e)}")


@app.post("/batch_add_tasks", include_in_schema=False)
def api_batch_add_tasks(req: BatchAddTasksRequest):
    added = []
    for item in req.tasks:
        res = core_add_task(title=item.title, due_date=item.due_date)
        added.append(res["task"])
    return {"success": True, "count": len(added), "tasks": added}


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
