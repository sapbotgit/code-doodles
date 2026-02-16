const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.json());
app.use(express.static('public'));

// In-memory storage
const chats = new Map();
const users = new Map();

// Translations
const translations = {
  en: {
    title: 'Modern Messenger',
    setUsername: 'Set Username',
    usernamePlaceholder: 'Your username',
    setBtn: 'Set',
    noChats: 'No chats yet. Create one!',
    enterUsernameToSee: 'Enter username to see chats',
    newChatName: 'New chat name',
    create: 'Create',
    privateChat: 'Private chat (requires password)',
    passwordPlaceholder: 'Set password',
    welcome: 'Welcome to Modern Messenger',
    welcomeDesc: 'Set your username and create or join a chat to start messaging',
    online: 'online',
    joined: 'Joined',
    join: 'Join',
    joinLock: 'Join 🔒',
    typeMessage: 'Type a message...',
    send: 'Send',
    privateChatPrompt: 'This is a private chat. Enter password:',
    incorrectPassword: 'Incorrect password',
    chatNotFound: 'Chat not found',
    setUsernameFirst: 'Please set username first',
    setPasswordPrivate: 'Please set a password for private chat',
    enterUsername: 'Please enter a username',
    systemJoined: 'joined the chat',
    language: 'Language:',
    privateChatTitle: '🔒 Private Chat',
    privateChatDesc: 'This chat requires a password to join',
    enterPassword: 'Enter password',
    cancel: 'Cancel',
    joinChat: 'Join Chat',
    passwordRequired: 'Password is required'
  },
  ru: {
    title: 'Современный Мессенджер',
    setUsername: 'Установить имя',
    usernamePlaceholder: 'Ваше имя',
    setBtn: 'Установить',
    noChats: 'Нет чатов. Создайте первый!',
    enterUsernameToSee: 'Введите имя для просмотра чатов',
    newChatName: 'Название чата',
    create: 'Создать',
    privateChat: 'Приватный чат (нужен пароль)',
    passwordPlaceholder: 'Установить пароль',
    welcome: 'Добро пожаловать в Мессенджер',
    welcomeDesc: 'Установите имя и создайте или присоединитесь к чату',
    online: 'онлайн',
    joined: 'Присоединились',
    join: 'Войти',
    joinLock: 'Войти 🔒',
    typeMessage: 'Введите сообщение...',
    send: 'Отправить',
    privateChatPrompt: 'Это приватный чат. Введите пароль:',
    incorrectPassword: 'Неверный пароль',
    chatNotFound: 'Чат не найден',
    setUsernameFirst: 'Сначала установите имя',
    setPasswordPrivate: 'Установите пароль для приватного чата',
    enterUsername: 'Введите имя',
    systemJoined: 'присоединился к чату',
    language: 'Язык:',
    privateChatTitle: '🔒 Приватный чат',
    privateChatDesc: 'Для входа в этот чат требуется пароль',
    enterPassword: 'Введите пароль',
    cancel: 'Отмена',
    joinChat: 'Войти в чат',
    passwordRequired: 'Требуется пароль'
  }
};

// HTML/JS/CSS all in one
app.get('/', (req, res) => {
  res.send(`
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Modern Messenger</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .container {
      width: 100%;
      max-width: 1200px;
      height: 90vh;
      background: white;
      border-radius: 20px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      display: flex;
      overflow: hidden;
    }
    .sidebar {
      width: 300px;
      background: #f8f9fa;
      border-right: 1px solid #e9ecef;
      display: flex;
      flex-direction: column;
    }
    .sidebar-header {
      padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
    }
    .sidebar-header h1 { font-size: 1.5rem; margin-bottom: 10px; }
    .user-input {
      display: flex;
      gap: 10px;
    }
    .user-input input {
      flex: 1;
      padding: 8px 12px;
      border: none;
      border-radius: 20px;
      font-size: 14px;
    }
    .btn {
      padding: 8px 16px;
      border: none;
      border-radius: 20px;
      cursor: pointer;
      font-size: 14px;
      transition: all 0.3s;
    }
    .btn-primary {
      background: #667eea;
      color: white;
    }
    .btn-primary:hover { background: #5568d3; }
    .btn-success {
      background: #28a745;
      color: white;
    }
    .btn-success:hover { background: #218838; }
    .btn-secondary {
      background: #6c757d;
      color: white;
    }
    .btn-secondary:hover { background: #5a6268; }
    .chat-list {
      flex: 1;
      overflow-y: auto;
      padding: 10px;
    }
    .chat-item {
      padding: 15px;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.2s;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .chat-item:hover { background: #e9ecef; }
    .chat-item.active { background: #667eea; color: white; }
    .chat-item span { font-weight: 500; }
    .chat-item small { opacity: 0.7; font-size: 12px; }
    .private-badge {
      font-size: 10px;
      background: #dc3545;
      color: white;
      padding: 2px 6px;
      border-radius: 8px;
      margin-left: 5px;
    }
    .new-chat {
      padding: 15px;
      border-top: 1px solid #e9ecef;
    }
    .new-chat-input {
      display: flex;
      gap: 10px;
      margin-bottom: 10px;
    }
    .new-chat-input input {
      flex: 1;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 20px;
      font-size: 14px;
    }
    .private-option {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: #666;
    }
    .private-option input[type="checkbox"] {
      width: 16px;
      height: 16px;
    }
    .main {
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    .chat-header {
      padding: 20px;
      border-bottom: 1px solid #e9ecef;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .chat-header h2 { color: #333; }
    .online-count { color: #28a745; font-size: 14px; }
    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      background: #fafafa;
    }
    .message {
      max-width: 70%;
      margin-bottom: 15px;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message.own { margin-left: auto; }
    .message-bubble {
      padding: 12px 16px;
      border-radius: 18px;
      word-wrap: break-word;
    }
    .message:not(.own) .message-bubble {
      background: white;
      border: 1px solid #e9ecef;
      border-bottom-left-radius: 4px;
    }
    .message.own .message-bubble {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .message-info {
      font-size: 12px;
      color: #888;
      margin-top: 4px;
      display: flex;
      gap: 8px;
    }
    .message.own .message-info { justify-content: flex-end; }
    .input-area {
      padding: 20px;
      border-top: 1px solid #e9ecef;
      display: flex;
      gap: 10px;
    }
    .input-area input {
      flex: 1;
      padding: 12px 20px;
      border: 1px solid #ddd;
      border-radius: 25px;
      font-size: 16px;
      outline: none;
    }
    .input-area input:focus { border-color: #667eea; }
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: #888;
    }
    .empty-state h3 { margin-bottom: 10px; color: #555; }
    .hidden { display: none !important; }
    .join-btn {
      padding: 4px 12px;
      background: #28a745;
      color: white;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      font-size: 12px;
    }
    .joined-badge {
      padding: 4px 12px;
      background: #6c757d;
      color: white;
      border-radius: 12px;
      font-size: 12px;
    }
    .lang-selector {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      font-size: 12px;
    }
    .lang-selector select {
      padding: 4px 8px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.3);
      background: rgba(255,255,255,0.2);
      color: white;
      cursor: pointer;
    }
    .lang-selector select option {
      color: #333;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="sidebar">
      <div class="sidebar-header">
        <div class="lang-selector">
          <span id="langLabel">Language:</span>
          <select id="langSelect">
            <option value="en">English</option>
            <option value="ru">Русский</option>
          </select>
        </div>
        <h1>💬 Messenger</h1>
        <div class="user-input">
          <input type="text" id="usernameInput" placeholder="Your username" maxlength="20">
          <button class="btn btn-primary" id="setUsernameBtn">Set</button>
        </div>
      </div>
      <div class="chat-list" id="chatList">
        <div style="text-align: center; padding: 20px; color: #888;" id="noChatsMsg">
          Enter username to see chats
        </div>
      </div>
      <div class="new-chat">
        <div class="new-chat-input">
          <input type="text" id="newChatName" placeholder="New chat name" maxlength="30">
          <button class="btn btn-success" id="createChatBtn">Create</button>
        </div>
        <label class="private-option">
          <input type="checkbox" id="isPrivate">
          <span id="privateChatLabel">Private chat (requires password)</span>
        </label>
        <div class="new-chat-input" id="passwordField" style="display: none; margin-top: 10px;">
          <input type="password" id="chatPassword" placeholder="Set password" maxlength="30">
        </div>
      </div>
    </div>
    <div class="main" id="mainArea">
      <div class="empty-state">
        <h3 id="welcomeTitle">Welcome to Modern Messenger</h3>
        <p id="welcomeDesc">Set your username and create or join a chat to start messaging</p>
      </div>
    </div>
  </div>

  <script src="/socket.io/socket.io.js"></script>
  <script>
    (function() {
      const socket = io();
      let currentUser = '';
      let currentChat = '';
      let joinedChats = new Set();
      let chatCache = {};
      let currentLang = 'en';

      const translations = ${JSON.stringify(translations)};

      function t(key) {
        return translations[currentLang][key] || translations['en'][key] || key;
      }

      function updateLanguage() {
        // Update static elements
        document.getElementById('langLabel').textContent = t('language');
        document.getElementById('usernameInput').placeholder = t('usernamePlaceholder');
        document.getElementById('setUsernameBtn').textContent = t('setBtn');
        document.getElementById('noChatsMsg').textContent = t('enterUsernameToSee');
        document.getElementById('newChatName').placeholder = t('newChatName');
        document.getElementById('createChatBtn').textContent = t('create');
        document.getElementById('privateChatLabel').textContent = t('privateChat');
        document.getElementById('chatPassword').placeholder = t('passwordPlaceholder');
        document.getElementById('welcomeTitle').textContent = t('welcome');
        document.getElementById('welcomeDesc').textContent = t('welcomeDesc');
        
        // Refresh chat list if visible
        if (currentUser) {
          loadChats();
        }
        
        // Refresh current chat if open
        if (currentChat && joinedChats.has(currentChat)) {
          selectChat(currentChat);
        }
      }

      function setUsername() {
        const usernameInput = document.getElementById('usernameInput');
        const username = usernameInput.value.trim();
        if (!username) return alert(t('enterUsername'));
        currentUser = username;
        socket.emit('set-username', username);
        usernameInput.disabled = true;
        document.getElementById('setUsernameBtn').disabled = true;
        loadChats();
      }

      function loadChats() {
        fetch('/chats')
          .then(function(r) { return r.json(); })
          .then(function(data) {
            chatCache = {};
            data.chats.forEach(function(c) { chatCache[c.id] = c; });
            renderChatList(data.chats);
          });
      }

      function renderChatList(chats) {
        const list = document.getElementById('chatList');
        if (chats.length === 0) {
          list.innerHTML = '<div style="text-align: center; padding: 20px; color: #888;">' + t('noChats') + '</div>';
          return;
        }
        list.innerHTML = '';
        for (let i = 0; i < chats.length; i++) {
          const chat = chats[i];
          const div = document.createElement('div');
          div.className = 'chat-item' + (currentChat === chat.id ? ' active' : '');
          
          const privateIcon = chat.isPrivate ? '<span class="private-badge">🔒</span>' : '';
          let joinBtnText = '';
          if (joinedChats.has(chat.id)) {
            joinBtnText = t('joined');
          } else if (chat.isPrivate) {
            joinBtnText = t('joinLock');
          } else {
            joinBtnText = t('join');
          }
          
          const joinBtnClass = joinedChats.has(chat.id) ? 'joined-badge' : 'join-btn';
          const joinBtnStyle = chat.isPrivate && !joinedChats.has(chat.id) ? ' style="background: #dc3545;"' : '';
          
          div.innerHTML = '<div><span>#' + chat.name + privateIcon + '</span><br><small>' + chat.userCount + ' ' + t('online') + '</small></div>' +
            '<button class="' + joinBtnClass + '"' + joinBtnStyle + '>' + joinBtnText + '</button>';
          
          div.onclick = function() {
            selectChat(chat.id);
          };
          
          const btn = div.querySelector('button');
          if (btn && !joinedChats.has(chat.id)) {
            btn.onclick = function(e) {
              e.stopPropagation();
              if (chat.isPrivate) {
                const password = prompt(t('privateChatPrompt'));
                if (password !== null) {
                  joinChat(chat.id, password);
                }
              } else {
                joinChat(chat.id, '');
              }
            };
          }
          
          list.appendChild(div);
        }
      }

      function togglePasswordField() {
        const isChecked = document.getElementById('isPrivate').checked;
        document.getElementById('passwordField').style.display = isChecked ? 'flex' : 'none';
      }

      function createChat() {
        if (!currentUser) return alert(t('setUsernameFirst'));
        const nameInput = document.getElementById('newChatName');
        const name = nameInput.value.trim();
        if (!name) return;
        const isPrivate = document.getElementById('isPrivate').checked;
        const password = isPrivate ? document.getElementById('chatPassword').value : '';
        if (isPrivate && !password) return alert(t('setPasswordPrivate'));
        socket.emit('create-chat', { name: name, isPrivate: isPrivate, password: password });
        nameInput.value = '';
        document.getElementById('isPrivate').checked = false;
        document.getElementById('chatPassword').value = '';
        document.getElementById('passwordField').style.display = 'none';
      }

      function joinChat(chatId, password) {
        if (!currentUser) return alert(t('setUsernameFirst'));
        socket.emit('join-chat', { chatId: chatId, password: password }, function(response) {
          if (response && response.success) {
            joinedChats.add(chatId);
            selectChat(chatId);
          } else {
            alert(response && response.error ? response.error : t('incorrectPassword'));
          }
        });
      }

      function selectChat(chatId) {
        if (!joinedChats.has(chatId)) return;
        currentChat = chatId;
        loadChats();
        fetch('/messages/' + chatId)
          .then(function(r) { return r.json(); })
          .then(function(data) { renderChat(data.messages); });
      }

      function renderChat(messages) {
        const chat = chatCache[currentChat];
        const chatName = chat ? chat.name : 'Chat';
        const privateLabel = chat && chat.isPrivate ? ' <span style="color: #dc3545;">🔒</span>' : '';
        
        let messagesHtml = '';
        for (let i = 0; i < messages.length; i++) {
          messagesHtml += renderMessage(messages[i]);
        }
        
        const mainArea = document.getElementById('mainArea');
        mainArea.innerHTML = 
          '<div class="chat-header">' +
            '<h2>#' + chatName + privateLabel + '</h2>' +
            '<span class="online-count">● ' + t('online') + '</span>' +
          '</div>' +
          '<div class="messages" id="messages">' + messagesHtml + '</div>' +
          '<div class="input-area">' +
            '<input type="text" id="messageInput" placeholder="' + t('typeMessage') + '">' +
            '<button class="btn btn-primary" id="sendBtn">' + t('send') + '</button>' +
          '</div>';
        
        const msgInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        
        sendBtn.onclick = function() { sendMessage(); };
        msgInput.onkeypress = function(e) {
          if (e.key === 'Enter') sendMessage();
        };
        msgInput.onkeydown = function(e) {
          if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
          }
        };
        
        scrollToBottom();
        msgInput.focus();
      }

      function renderMessage(msg) {
        const isOwn = msg.username === currentUser;
        let username = msg.username;
        if (username === 'System') {
          username = currentLang === 'ru' ? 'Система' : 'System';
        }
        return '<div class="message ' + (isOwn ? 'own' : '') + '">' +
          '<div class="message-bubble">' + escapeHtml(msg.text) + '</div>' +
          '<div class="message-info">' +
            '<span>' + escapeHtml(username) + '</span>' +
            '<span>' + new Date(msg.timestamp).toLocaleTimeString() + '</span>' +
          '</div>' +
        '</div>';
      }

      function sendMessage() {
        const input = document.getElementById('messageInput');
        if (!input) return;
        const text = input.value.trim();
        if (!text || !currentChat) return;
        socket.emit('send-message', { chatId: currentChat, text: text });
        input.value = '';
      }

      function scrollToBottom() {
        const messages = document.getElementById('messages');
        if (messages) messages.scrollTop = messages.scrollHeight;
      }

      function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
      }

      // Setup event listeners
      document.getElementById('langSelect').onchange = function() {
        currentLang = this.value;
        updateLanguage();
      };
      
      document.getElementById('setUsernameBtn').onclick = setUsername;
      document.getElementById('createChatBtn').onclick = createChat;
      document.getElementById('isPrivate').onchange = togglePasswordField;
      document.getElementById('usernameInput').onkeypress = function(e) {
        if (e.key === 'Enter') setUsername();
      };
      document.getElementById('newChatName').onkeypress = function(e) {
        if (e.key === 'Enter') createChat();
      };

      // Socket events
      socket.on('chat-created', loadChats);
      socket.on('user-joined', loadChats);
      socket.on('user-left', loadChats);
      
      socket.on('new-message', function(data) {
        if (data.chatId === currentChat) {
          const messages = document.getElementById('messages');
          if (messages) {
            messages.innerHTML += renderMessage(data.message);
            scrollToBottom();
          }
        }
      });

      // Refresh chat list periodically
      setInterval(function() {
        if (currentUser) loadChats();
      }, 3000);
    })();
  </script>
</body>
</html>
  `);
});

// API Routes
app.get('/chats', (req, res) => {
  const chatList = Array.from(chats.values()).map(chat => ({
    id: chat.id,
    name: chat.name,
    userCount: chat.users.size,
    isPrivate: chat.isPrivate
  }));
  res.json({ chats: chatList });
});

app.get('/messages/:chatId', (req, res) => {
  const chat = chats.get(req.params.chatId);
  if (!chat) return res.json({ messages: [] });
  res.json({ messages: chat.messages });
});

// Socket.io handling
io.on('connection', (socket) => {
  let username = '';
  let joinedChats = new Set();

  socket.on('set-username', (name) => {
    username = name;
    users.set(socket.id, { username, socket });
  });

  socket.on('create-chat', ({ name, isPrivate, password }) => {
    const chatId = 'chat_' + Date.now();
    const chat = {
      id: chatId,
      name: name,
      isPrivate: isPrivate || false,
      password: password || null,
      users: new Set(),
      messages: []
    };
    chats.set(chatId, chat);
    io.emit('chat-created');
  });

  socket.on('join-chat', ({ chatId, password }, callback) => {
    const chat = chats.get(chatId);
    if (!chat) {
      return callback({ success: false, error: 'Chat not found' });
    }
    
    // Check password for private chats
    if (chat.isPrivate && chat.password !== password) {
      return callback({ success: false, error: 'Incorrect password' });
    }
    
    socket.join(chatId);
    chat.users.add(socket.id);
    joinedChats.add(chatId);
    
    socket.to(chatId).emit('user-joined');
    
    // Send join notification
    const joinMsg = {
      username: 'System',
      text: username + ' joined the chat',
      timestamp: Date.now()
    };
    chat.messages.push(joinMsg);
    io.to(chatId).emit('new-message', { chatId, message: joinMsg });
    
    callback({ success: true });
  });

  socket.on('send-message', ({ chatId, text }) => {
    const chat = chats.get(chatId);
    if (!chat || !joinedChats.has(chatId)) return;
    
    const message = {
      username,
      text,
      timestamp: Date.now()
    };
    chat.messages.push(message);
    io.to(chatId).emit('new-message', { chatId, message });
  });

  socket.on('disconnect', () => {
    users.delete(socket.id);
    joinedChats.forEach(chatId => {
      const chat = chats.get(chatId);
      if (chat) {
        chat.users.delete(socket.id);
        socket.to(chatId).emit('user-left');
      }
    });
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log('🚀 Messenger running on http://localhost:' + PORT);
});
