// ==UserScript==
// @name         Pandy-ComfyUI Token Grabber
// @namespace    http://tampermonkey.net/
// @version      3.5
// @description  直接从 ComfyUI 提取 auth_token_comfy_org 并同步到 Pandy
// @author       You
// @match        *://localhost:8188/*
// @match        *://127.0.0.1:8188/*
// @match        *://127.0.0.1:8180/*
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_setClipboard
// @grant        GM_xmlhttpRequest
// @grant        unsafeWindow
// @connect      127.0.0.1
// @connect      localhost
// @run-at       document-idle
// ==/UserScript==

(function () {
    'use strict';

    const STORAGE_KEY = 'comfyui_auth_token';
    const POS_KEY = 'token_grabber_pos';
    const COLLAPSED_KEY = 'token_grabber_collapsed';

    let foundToken = '';
    let isCollapsed = GM_getValue(COLLAPSED_KEY, false);
    let savedPos = GM_getValue(POS_KEY, { x: null, y: 10 });

    // 确保位置在可视范围内
    function clampPosition() {
        const panel = document.getElementById('token-grabber-panel');
        if (!panel) return;

        const rect = panel.getBoundingClientRect();
        let needUpdate = false;

        // 如果面板超出右边界
        if (rect.right > window.innerWidth) {
            panel.style.left = 'auto';
            panel.style.right = '10px';
            needUpdate = true;
        }
        // 如果面板超出下边界
        if (rect.bottom > window.innerHeight) {
            panel.style.top = Math.max(10, window.innerHeight - rect.height - 10) + 'px';
            needUpdate = true;
        }
        // 如果面板超出左边界
        if (rect.left < 0) {
            panel.style.left = '10px';
            panel.style.right = 'auto';
            needUpdate = true;
        }
        // 如果面板超出上边界
        if (rect.top < 0) {
            panel.style.top = '10px';
            needUpdate = true;
        }
    }

    // 搜索 token
    function searchToken() {
        // 1. localStorage
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            const value = localStorage.getItem(key);
            if (value && value.includes('eyJ')) {
                const token = extractJWT(value);
                if (token && isValidJWT(token)) return token;
            }
        }
        // 2. sessionStorage
        for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i);
            const value = sessionStorage.getItem(key);
            if (value && value.includes('eyJ')) {
                const token = extractJWT(value);
                if (token && isValidJWT(token)) return token;
            }
        }
        // 3. window 对象
        const props = ['app', 'api', 'comfyAPI'];
        for (const prop of props) {
            if (window[prop]) {
                const token = deepSearch(window[prop], 'auth_token_comfy_org', 5);
                if (token) return token;
            }
        }
        return null;
    }

    function extractJWT(str) {
        const match = str.match(/eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/);
        return match ? match[0] : null;
    }

    function isValidJWT(token) {
        try {
            const parts = token.split('.');
            if (parts.length !== 3) return false;
            let payload = parts[1];
            payload += '='.repeat((4 - payload.length % 4) % 4);
            const info = JSON.parse(atob(payload));
            return info.exp !== undefined;
        } catch (e) { return false; }
    }

    function deepSearch(obj, key, depth) {
        if (depth <= 0 || !obj || typeof obj !== 'object') return null;
        try {
            if (obj[key]) return obj[key];
            for (const k in obj) {
                const result = deepSearch(obj[k], key, depth - 1);
                if (result) return result;
            }
        } catch (e) { }
        return null;
    }

    function parseToken(token) {
        try {
            const parts = token.split('.');
            let payload = parts[1];
            payload += '='.repeat((4 - payload.length % 4) % 4);
            const info = JSON.parse(atob(payload));
            const remaining = Math.floor((info.exp * 1000 - Date.now()) / 60000);
            return { valid: remaining > 0, remaining };
        } catch (e) { return { valid: false, remaining: 0 }; }
    }

    function updatePanel() {
        const statusEl = document.getElementById('tg-status');
        const previewEl = document.getElementById('tg-preview');
        const expiryEl = document.getElementById('tg-expiry');
        const iconStatus = document.getElementById('tg-icon-status');

        if (!foundToken) {
            if (statusEl) {
                statusEl.textContent = '❌ 未找到';
                statusEl.className = 'status expired';
            }
            if (previewEl) previewEl.textContent = '请确保已登录';
            if (iconStatus) iconStatus.style.background = '#f66';
            return;
        }

        const info = parseToken(foundToken);
        if (info.valid) {
            if (statusEl) {
                statusEl.textContent = '✅ Token 已获取';
                statusEl.className = 'status captured';
            }
            if (expiryEl) expiryEl.innerHTML = `⏱️ 剩余 <b>${info.remaining}</b> 分钟`;
            if (iconStatus) iconStatus.style.background = '#6f6';
        } else {
            if (statusEl) {
                statusEl.textContent = '❌ Token 已过期';
                statusEl.className = 'status expired';
            }
            if (expiryEl) expiryEl.innerHTML = '请重新登录';
            if (iconStatus) iconStatus.style.background = '#f66';
        }
        if (previewEl) previewEl.textContent = foundToken.substring(0, 50) + '...';
    }

    function copyToken() {
        if (foundToken) {
            GM_setClipboard(foundToken);
            const btn = document.getElementById('tg-copy-btn');
            if (btn) {
                btn.textContent = '✅ 已复制!';
                setTimeout(() => btn.textContent = '📋 复制 Token', 1500);
            }
        } else {
            alert('未找到 Token');
        }
    }

    function refreshToken() {
        foundToken = searchToken() || '';
        if (foundToken) GM_setValue(STORAGE_KEY, foundToken);
        updatePanel();
    }

    function toggleCollapse() {
        isCollapsed = !isCollapsed;
        GM_setValue(COLLAPSED_KEY, isCollapsed);
        const panel = document.getElementById('token-grabber-panel');
        if (isCollapsed) {
            panel.classList.add('collapsed');
        } else {
            panel.classList.remove('collapsed');
        }
    }

    // 同步到 Pandy Image 页面（使用 GM_xmlhttpRequest 绕过跨域）
    async function syncToPandy() {
        if (!foundToken) {
            alert('未找到 Token');
            return;
        }

        const btn = document.getElementById('tg-sync-btn');
        if (btn) btn.textContent = '⏳ 同步中...';

        const PANDY_BASE = 'http://127.0.0.1:8180';

        // 使用 GM_xmlhttpRequest 绕过跨域限制
        // 1. 先获取现有配置
        GM_xmlhttpRequest({
            method: 'GET',
            url: `${PANDY_BASE}/config?name=settings`,
            onload: function (response) {
                let config = {};
                try {
                    if (response.status === 200) {
                        config = JSON.parse(response.responseText);
                    }
                } catch (e) { }

                // 2. 合并 token
                config.token = foundToken;

                // 3. POST 回去
                GM_xmlhttpRequest({
                    method: 'POST',
                    url: `${PANDY_BASE}/config?name=settings`,
                    headers: { 'Content-Type': 'application/json' },
                    data: JSON.stringify(config),
                    onload: function (resp) {
                        if (resp.status === 200) {
                            if (btn) {
                                btn.textContent = '✅ 已同步!';
                                setTimeout(() => btn.textContent = '🚀 同步到 Pandy', 2000);
                            }
                            console.log('🍌 Token 已同步到 Pandy');
                        } else {
                            fallbackCopy(btn);
                        }
                    },
                    onerror: function () {
                        fallbackCopy(btn);
                    }
                });
            },
            onerror: function () {
                fallbackCopy(btn);
            }
        });
    }

    function fallbackCopy(btn) {
        GM_setClipboard(foundToken);
        if (btn) {
            btn.textContent = '❌ 服务未启动，已复制';
            setTimeout(() => btn.textContent = '🚀 同步到 Pandy', 3000);
        }
        console.log('🍌 同步失败，请确保 Pandy 服务器已启动 (python server.py)');
    }

    unsafeWindow.copyToken = copyToken;
    unsafeWindow.refreshToken = refreshToken;
    unsafeWindow.toggleCollapse = toggleCollapse;
    unsafeWindow.syncToPandy = syncToPandy;

    function createPanel() {
        const panel = document.createElement('div');
        panel.id = 'token-grabber-panel';
        if (isCollapsed) panel.classList.add('collapsed');

        // 设置位置
        if (savedPos.x !== null) {
            panel.style.left = savedPos.x + 'px';
            panel.style.right = 'auto';
        }
        panel.style.top = savedPos.y + 'px';

        panel.innerHTML = `
            <style>
                #token-grabber-panel {
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    background: #1a1a2e;
                    border: 2px solid #f0c040;
                    border-radius: 10px;
                    padding: 12px 15px;
                    z-index: 999999;
                    font-family: sans-serif;
                    font-size: 13px;
                    color: #fff;
                    width: 260px;
                    box-shadow: 0 5px 20px rgba(0,0,0,0.5);
                    transition: width 0.2s, padding 0.2s, border-radius 0.2s;
                    cursor: default;
                }
                #token-grabber-panel .header {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    cursor: move;
                    user-select: none;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #333;
                    margin-bottom: 10px;
                }
                #token-grabber-panel .icon {
                    font-size: 20px;
                }
                #token-grabber-panel .title {
                    flex: 1;
                    font-weight: bold;
                    color: #f0c040;
                    font-size: 14px;
                }
                #token-grabber-panel .icon-status {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: #666;
                }
                #token-grabber-panel .collapse-btn {
                    cursor: pointer;
                    color: #888;
                    font-size: 16px;
                    padding: 2px 6px;
                    border-radius: 4px;
                }
                #token-grabber-panel .collapse-btn:hover {
                    background: #333;
                    color: #fff;
                }
                #token-grabber-panel .content {
                    transition: opacity 0.2s;
                }
                #token-grabber-panel .status {
                    padding: 6px 10px;
                    border-radius: 6px;
                    margin: 8px 0;
                    font-weight: bold;
                    font-size: 12px;
                }
                #token-grabber-panel .status.captured { background: rgba(50,200,50,0.2); color: #6f6; }
                #token-grabber-panel .status.expired { background: rgba(200,50,50,0.2); color: #f66; }
                #token-grabber-panel .token-preview {
                    font-family: monospace;
                    font-size: 9px;
                    color: #666;
                    word-break: break-all;
                    background: #111;
                    padding: 6px;
                    border-radius: 4px;
                    margin: 8px 0;
                }
                #token-grabber-panel #tg-expiry {
                    color: #6f6;
                    font-size: 12px;
                    margin: 6px 0;
                }
                #token-grabber-panel button {
                    background: #f0c040;
                    color: #000;
                    border: none;
                    padding: 8px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: bold;
                    width: 100%;
                    margin-top: 4px;
                    font-size: 12px;
                }
                #token-grabber-panel button:hover { background: #e6a000; }
                #token-grabber-panel button.secondary {
                    background: #333;
                    color: #aaa;
                }
                
                /* 收缩状态 */
                #token-grabber-panel.collapsed {
                    width: auto;
                    padding: 8px 12px;
                    border-radius: 25px;
                }
                #token-grabber-panel.collapsed .header {
                    padding-bottom: 0;
                    border-bottom: none;
                    margin-bottom: 0;
                }
                #token-grabber-panel.collapsed .content {
                    display: none;
                }
                #token-grabber-panel.collapsed .title {
                    display: none;
                }
            </style>
            <div class="header" id="tg-header">
                <span class="icon">🍌</span>
                <span class="title">Token Grabber</span>
                <span class="icon-status" id="tg-icon-status"></span>
                <span class="collapse-btn" onclick="toggleCollapse()" title="收缩/展开">▼</span>
            </div>
            <div class="content">
                <div class="status" id="tg-status">🔍 搜索中...</div>
                <div class="token-preview" id="tg-preview">正在搜索...</div>
                <div id="tg-expiry"></div>
                <button id="tg-copy-btn" onclick="copyToken()">📋 复制 Token</button>
                <button id="tg-sync-btn" onclick="syncToPandy()">🚀 同步到 Pandy</button>
                <button class="secondary" onclick="refreshToken()">🔄 刷新</button>
            </div>
        `;
        document.body.appendChild(panel);

        // 拖拽功能
        const header = document.getElementById('tg-header');
        let isDragging = false;
        let startX, startY, startLeft, startTop;

        header.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('collapse-btn')) return;
            isDragging = true;
            const rect = panel.getBoundingClientRect();
            startX = e.clientX;
            startY = e.clientY;
            startLeft = rect.left;
            startTop = rect.top;
            panel.style.right = 'auto';
            panel.style.left = startLeft + 'px';
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            let newLeft = startLeft + dx;
            let newTop = startTop + dy;

            // 边界限制
            newLeft = Math.max(0, Math.min(newLeft, window.innerWidth - panel.offsetWidth));
            newTop = Math.max(0, Math.min(newTop, window.innerHeight - panel.offsetHeight));

            panel.style.left = newLeft + 'px';
            panel.style.top = newTop + 'px';
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                // 保存位置
                const rect = panel.getBoundingClientRect();
                savedPos = { x: rect.left, y: rect.top };
                GM_setValue(POS_KEY, savedPos);
            }
        });
    }

    function init() {
        console.log('🍌 Token Grabber: init() 开始执行');
        console.log('🍌 document.body:', document.body);

        if (!document.body) {
            console.log('🍌 body 不存在，延迟重试...');
            setTimeout(init, 500);
            return;
        }

        createPanel();
        console.log('🍌 面板已创建');

        // 确保面板在可视范围内
        setTimeout(clampPosition, 100);

        // 监听窗口大小变化
        window.addEventListener('resize', clampPosition);

        setTimeout(() => {
            foundToken = searchToken() || GM_getValue(STORAGE_KEY, '');
            if (foundToken) GM_setValue(STORAGE_KEY, foundToken);
            updatePanel();
            console.log('🍌 Token 搜索完成:', foundToken ? '找到' : '未找到');
        }, 1500);

        setInterval(() => {
            const newToken = searchToken();
            if (newToken && newToken !== foundToken) {
                foundToken = newToken;
                GM_setValue(STORAGE_KEY, foundToken);
                updatePanel();
            }
        }, 10000);
    }

    // 多种方式确保执行
    console.log('🍌 Token Grabber: 脚本已加载, readyState:', document.readyState);

    if (document.readyState === 'complete') {
        init();
    } else if (document.readyState === 'interactive') {
        setTimeout(init, 100);
    } else {
        window.addEventListener('load', init);
        document.addEventListener('DOMContentLoaded', () => setTimeout(init, 100));
    }
})();
