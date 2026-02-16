const express = require('express');
const fs = require('fs');
const path = require('path');

const router = express.Router();

const CONFIG = {
  ACCESS_FOLDER: './files',
  ADMIN_KEY: 'key',
  KEYS_FILE: 'enc_keys.json'
};

let keysData = {
  keys: {},
  traffic: {}
};

function loadKeys() {
  try {
    if (fs.existsSync(CONFIG.KEYS_FILE)) {
      const data = fs.readFileSync(CONFIG.KEYS_FILE, 'utf8');
      keysData = JSON.parse(data);
      if (!keysData.traffic) keysData.traffic = {};
    }
  } catch (err) {
    console.error('Error loading keys:', err);
  }
}

function saveKeys() {
  try {
    fs.writeFileSync(CONFIG.KEYS_FILE, JSON.stringify(keysData, null, 2));
  } catch (err) {
    console.error('Error saving keys:', err);
  }
}

loadKeys();

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function generateKey() {
  return Array.from({ length: 32 }, () => 
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    [Math.floor(Math.random() * 62)]
  ).join('');
}

function requireAdmin(req, res, next) {
  const adminKey = req.headers['x-admin-key'] || req.query.admin_key;
  if (adminKey !== CONFIG.ADMIN_KEY) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
}

function requireAccessKey(req, res, next) {
  const accessKey = req.headers['x-access-key'] || req.query.access_key;
  if (!accessKey || !keysData.keys[accessKey]) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  req.accessKey = accessKey;
  next();
}

const commonStyles = `
:root {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --accent-primary: #3b82f6;
  --accent-secondary: #8b5cf6;
  --accent-success: #10b981;
  --accent-danger: #ef4444;
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --border: #475569;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: linear-gradient(135deg, var(--bg-primary) 0%, #1a1a2e 100%);
  color: var(--text-primary);
  min-height: 100vh;
  line-height: 1.6;
}
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
.header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 20px 0;
  margin-bottom: 30px;
}
.header-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.logo {
  font-size: 24px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  transition: all 0.3s ease;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
}
.btn-primary {
  background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
  color: white;
}
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4); }
.btn-success { background: var(--accent-success); color: white; }
.btn-success:hover { background: #059669; transform: translateY(-2px); }
.btn-danger { background: var(--accent-danger); color: white; }
.btn-danger:hover { background: #dc2626; transform: translateY(-2px); }
.btn-secondary { background: var(--bg-tertiary); color: var(--text-primary); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); }
.card {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 20px;
  border: 1px solid var(--border);
}
.card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.card-title { font-size: 18px; font-weight: 600; }
input[type="text"], input[type="password"] {
  width: 100%;
  padding: 12px 16px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text-primary);
  font-size: 14px;
}
input[type="text"]:focus, input[type="password"]:focus {
  outline: none;
  border-color: var(--accent-primary);
}
.input-group { display: flex; gap: 10px; margin-bottom: 15px; }
.input-group input { flex: 1; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 14px 16px; text-align: left; border-bottom: 1px solid var(--border); }
th { font-weight: 600; color: var(--text-secondary); font-size: 12px; text-transform: uppercase; }
td { font-size: 14px; }
tr:hover td { background: var(--bg-tertiary); }
.key-display {
  font-family: 'Courier New', monospace;
  background: var(--bg-tertiary);
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
  color: var(--accent-primary);
}
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
.stat-card {
  background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
  padding: 20px;
  border-radius: 12px;
  border: 1px solid var(--border);
}
.stat-label { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 8px; }
.stat-value { font-size: 28px; font-weight: 700; }
.breadcrumb {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 20px;
  padding: 12px 16px;
  background: var(--bg-tertiary);
  border-radius: 8px;
  font-size: 14px;
}
.breadcrumb a { color: var(--accent-primary); text-decoration: none; }
.breadcrumb-separator { color: var(--text-secondary); }
.file-list { display: flex; flex-direction: column; gap: 8px; }
.file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  background: var(--bg-tertiary);
  border-radius: 8px;
  transition: all 0.2s ease;
}
.file-item:hover { background: var(--border); transform: translateX(4px); }
.file-info { display: flex; align-items: center; gap: 12px; }
.file-icon { font-size: 20px; }
.file-name { font-weight: 500; }
.file-size { font-size: 12px; color: var(--text-secondary); }
.empty-state { text-align: center; padding: 60px 20px; color: var(--text-secondary); }
.empty-state-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.5; }
.toast {
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 16px 24px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  display: none;
  z-index: 1000;
}
.toast.show { display: flex; }
.landing-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 30px;
  margin-top: 40px;
}
.landing-card {
  background: var(--bg-secondary);
  border-radius: 16px;
  padding: 40px;
  text-align: center;
  border: 1px solid var(--border);
  transition: all 0.3s ease;
  cursor: pointer;
}
.landing-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.3);
  border-color: var(--accent-primary);
}
.landing-icon { font-size: 64px; margin-bottom: 20px; }
.landing-title { font-size: 24px; font-weight: 700; margin-bottom: 12px; }
.landing-desc { color: var(--text-secondary); margin-bottom: 24px; }
.modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal-overlay.show { display: flex; }
.modal {
  background: var(--bg-secondary);
  border-radius: 16px;
  padding: 32px;
  width: 90%;
  max-width: 400px;
  border: 1px solid var(--border);
}
.modal-title { font-size: 20px; font-weight: 600; margin-bottom: 20px; }
.badge {
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  background: rgba(16, 185, 129, 0.2);
  color: var(--accent-success);
}
`;

router.get('/', (req, res) => {
  res.send(`<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Encrypted Access Portal</title>
  <style>${commonStyles}</style>
</head>
<body>
  <div class="header">
    <div class="header-content">
      <div class="logo">ENCRYPTED ACCESS</div>
    </div>
  </div>
  <div class="container">
    <div class="landing-grid">
      <div class="landing-card" onclick="openModal('admin')">
        <div class="landing-icon">🔐</div>
        <div class="landing-title">Admin Panel</div>
        <div class="landing-desc">Manage access keys, view traffic statistics, and control user permissions</div>
        <button class="btn btn-primary">Enter Admin</button>
      </div>
      <div class="landing-card" onclick="openModal('access')">
        <div class="landing-icon">📁</div>
        <div class="landing-title">Access Panel</div>
        <div class="landing-desc">Browse files and download content with your access key</div>
        <button class="btn btn-success">Enter Access</button>
      </div>
    </div>
  </div>
  <div class="modal-overlay" id="modal">
    <div class="modal">
      <div class="modal-title" id="modalTitle">Enter Key</div>
      <input type="password" id="keyInput" placeholder="Enter your key..." onkeypress="if(event.key==='Enter')submitKey()">
      <div style="margin-top: 20px; display: flex; gap: 10px; justify-content: flex-end;">
        <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
        <button class="btn btn-primary" onclick="submitKey()">Enter</button>
      </div>
    </div>
  </div>
  <script>
    let currentType = '';
    function openModal(type) {
      currentType = type;
      document.getElementById('modalTitle').textContent = type === 'admin' ? 'Admin Key' : 'Access Key';
      document.getElementById('modal').classList.add('show');
      document.getElementById('keyInput').value = '';
      document.getElementById('keyInput').focus();
    }
    function closeModal() {
      document.getElementById('modal').classList.remove('show');
    }
    function submitKey() {
      const key = document.getElementById('keyInput').value.trim();
      if (!key) return;
      if (currentType === 'admin') {
        window.location.href = '/enc/admin?admin_key=' + encodeURIComponent(key);
      } else {
        window.location.href = '/enc/access?access_key=' + encodeURIComponent(key);
      }
    }
    document.getElementById('modal').addEventListener('click', function(e) {
      if (e.target === this) closeModal();
    });
  </script>
</body>
</html>`);
});

router.get('/admin', requireAdmin, (req, res) => {
  res.send(`<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin Panel</title>
  <style>${commonStyles}</style>
</head>
<body>
  <div class="header">
    <div class="header-content">
      <div class="logo">ADMIN PANEL</div>
      <div class="nav-links">
        <button class="btn btn-secondary" onclick="location.href='/enc/'">🏠 Home</button>
        <button class="btn btn-secondary" onclick="loadKeys()">🔄 Refresh</button>
      </div>
    </div>
  </div>
  <div class="container">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Total Keys</div>
        <div class="stat-value" id="totalKeys">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Traffic</div>
        <div class="stat-value" id="totalTraffic">0 B</div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">
        <div class="card-title">Create New Access Key</div>
      </div>
      <div class="input-group">
        <input type="text" id="newKeyName" placeholder="Key name/description (optional)">
        <button class="btn btn-success" onclick="createKey()">Create Key</button>
      </div>
      <div id="newKeyResult" style="display: none; margin-top: 15px; padding: 16px; background: var(--bg-tertiary); border-radius: 8px;">
        <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 8px;">New key created:</div>
        <div class="key-display" id="createdKey"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">
        <div class="card-title">Access Keys & Traffic</div>
      </div>
      <div id="keysTable">
        <div class="empty-state">
          <div class="empty-state-icon">🔑</div>
          <div>No keys created yet</div>
        </div>
      </div>
    </div>
  </div>
  <div class="toast" id="toast">
    <span id="toastMessage"></span>
  </div>
  <script>
    const adminKey = new URLSearchParams(window.location.search).get('admin_key');
    function showToast(message) {
      const toast = document.getElementById('toast');
      document.getElementById('toastMessage').textContent = message;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 3000);
    }
    function formatBytes(bytes) {
      if (!bytes || bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    async function loadKeys() {
      const res = await fetch('/enc/admin/keys?admin_key=' + adminKey);
      const data = await res.json();
      document.getElementById('totalKeys').textContent = Object.keys(data.keys).length;
      let totalBytes = 0;
      Object.values(data.traffic || {}).forEach(bytes => totalBytes += bytes);
      document.getElementById('totalTraffic').textContent = formatBytes(totalBytes);
      const keysCount = Object.keys(data.keys).length;
      if (keysCount === 0) {
        document.getElementById('keysTable').innerHTML = '<div class="empty-state"><div class="empty-state-icon">🔑</div><div>No keys created yet</div></div>';
        return;
      }
      let html = '<table><thead><tr><th>Key</th><th>Name</th><th>Created</th><th>Traffic</th><th>Actions</th></tr></thead><tbody>';
      for (const [key, info] of Object.entries(data.keys)) {
        const traffic = data.traffic?.[key] || 0;
        html += '<tr><td><div class="key-display">' + key.substring(0, 16) + '...</div></td>';
        html += '<td>' + (info.name || '-') + '</td>';
        html += '<td>' + new Date(info.created).toLocaleDateString() + '</td>';
        html += '<td><span class="badge">' + formatBytes(traffic) + '</span></td>';
        html += '<td><button class="btn btn-danger" style="padding: 6px 12px; font-size: 12px;" onclick="deleteKey(' + "'" + key + "'" + ')">Delete</button></td></tr>';
      }
      html += '</tbody></table>';
      document.getElementById('keysTable').innerHTML = html;
    }
    async function createKey() {
      const name = document.getElementById('newKeyName').value;
      const res = await fetch('/enc/admin/keys?admin_key=' + adminKey, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      const data = await res.json();
      document.getElementById('createdKey').textContent = data.key;
      document.getElementById('newKeyResult').style.display = 'block';
      document.getElementById('newKeyName').value = '';
      showToast('Key created successfully!');
      loadKeys();
    }
    async function deleteKey(key) {
      if (!confirm('Are you sure you want to delete this key?\\n\\nThis action cannot be undone.')) return;
      await fetch('/enc/admin/keys/' + key + '?admin_key=' + adminKey, { method: 'DELETE' });
      showToast('Key deleted successfully');
      loadKeys();
    }
    loadKeys();
  </script>
</body>
</html>`);
});

router.get('/admin/keys', requireAdmin, (req, res) => {
  res.json(keysData);
});

router.post('/admin/keys', requireAdmin, (req, res) => {
  const key = generateKey();
  const name = req.body?.name || '';
  keysData.keys[key] = { name: name, created: new Date().toISOString() };
  keysData.traffic[key] = 0;
  saveKeys();
  res.json({ key, message: 'Key created successfully' });
});

router.delete('/admin/keys/:key', requireAdmin, (req, res) => {
  const key = req.params.key;
  if (keysData.keys[key]) {
    delete keysData.keys[key];
    delete keysData.traffic[key];
    saveKeys();
    res.json({ message: 'Key deleted successfully' });
  } else {
    res.status(404).json({ error: 'Key not found' });
  }
});

router.get('/access', requireAccessKey, (req, res) => {
  res.send(`<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Access Panel</title>
  <style>${commonStyles}</style>
</head>
<body>
  <div class="header">
    <div class="header-content">
      <div class="logo">FILE ACCESS</div>
      <div class="nav-links">
        <button class="btn btn-secondary" onclick="location.href='/enc/'">🏠 Home</button>
      </div>
    </div>
  </div>
  <div class="container">
    <div class="card">
      <div class="card-header">
        <div class="card-title">Your Traffic</div>
      </div>
      <div style="display: flex; align-items: baseline; gap: 10px;">
        <span style="font-size: 36px; font-weight: 700; color: var(--accent-success);" id="trafficValue">0 B</span>
        <span style="color: var(--text-secondary);">downloaded</span>
      </div>
    </div>
    <div class="card">
      <div class="card-header">
        <div class="card-title">Files</div>
      </div>
      <div class="breadcrumb" id="breadcrumb">
        <a href="#" onclick="loadFiles(''); return false;">📁 Root</a>
      </div>
      <div id="fileList">
        <div class="empty-state">
          <div class="empty-state-icon">📂</div>
          <div>Loading...</div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const accessKey = new URLSearchParams(window.location.search).get('access_key');
    let currentPath = '';
    function formatBytes(bytes) {
      if (!bytes || bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    async function loadFiles(path) {
      if (path === undefined) path = '';
      currentPath = path;
      const res = await fetch('/enc/access/files?access_key=' + accessKey + '&path=' + encodeURIComponent(path));
      const data = await res.json();
      document.getElementById('trafficValue').textContent = formatBytes(data.traffic || 0);
      let breadcrumbHtml = '<a href="#" onclick="loadFiles(' + "''" + '); return false;">📁 Root</a>';
      if (path) {
        const parts = path.split('/').filter(function(p) { return p; });
        let builtPath = '';
        parts.forEach(function(part, i) {
          builtPath += '/' + part;
          breadcrumbHtml += '<span class="breadcrumb-separator">/</span><a href="#" onclick="loadFiles(' + "'" + builtPath + "'" + '); return false;">' + part + '</a>';
        });
      }
      document.getElementById('breadcrumb').innerHTML = breadcrumbHtml;
      if (data.files.length === 0) {
        document.getElementById('fileList').innerHTML = '<div class="empty-state"><div class="empty-state-icon">📂</div><div>This folder is empty</div></div>';
        return;
      }
      let html = '<div class="file-list">';
      data.files.forEach(function(file) {
        if (file.type === 'directory') {
          html += '<div class="file-item" onclick="loadFiles(' + "'" + file.path + "'" + ')" style="cursor: pointer;">';
          html += '<div class="file-info"><span class="file-icon">📁</span><div><div class="file-name">' + file.name + '</div><div class="file-size">Folder</div></div></div>';
          html += '<button class="btn btn-primary" style="padding: 8px 16px; font-size: 12px;">Open</button></div>';
        } else {
          html += '<div class="file-item">';
          html += '<div class="file-info"><span class="file-icon">📄</span><div><div class="file-name">' + file.name + '</div><div class="file-size">' + formatBytes(file.size) + '</div></div></div>';
          html += '<a href="/enc/access/download?access_key=' + accessKey + '&file=' + encodeURIComponent(file.path) + '" class="btn btn-success" style="padding: 8px 16px; font-size: 12px; text-decoration: none;" onclick="setTimeout(function() { loadFiles(currentPath); }, 500)">Download</a></div>';
        }
      });
      html += '</div>';
      document.getElementById('fileList').innerHTML = html;
    }
    loadFiles();
  </script>
</body>
</html>`);
});

router.get('/access/files', requireAccessKey, (req, res) => {
  const accessKey = req.accessKey;
  const relativePath = req.query.path || '';
  const sanitizedPath = path.normalize(relativePath).replace(/^(\.\.(\/|\\|$))+/, '');
  const fullPath = path.join(path.resolve(CONFIG.ACCESS_FOLDER), sanitizedPath);
  if (!fullPath.startsWith(path.resolve(CONFIG.ACCESS_FOLDER))) {
    return res.status(403).json({ error: 'Access denied' });
  }
  try {
    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ error: 'Path not found' });
    }
    const items = fs.readdirSync(fullPath);
    const files = items.map(function(item) {
      const itemPath = path.join(fullPath, item);
      const stat = fs.statSync(itemPath);
      const relativeItemPath = path.join(sanitizedPath, item).replace(/\\/g, '/');
      return {
        name: item,
        path: relativeItemPath,
        type: stat.isDirectory() ? 'directory' : 'file',
        size: stat.isFile() ? stat.size : null
      };
    });
    res.json({
      path: sanitizedPath,
      files: files,
      traffic: keysData.traffic[accessKey] || 0
    });
  } catch (err) {
    res.status(500).json({ error: 'Error reading directory' });
  }
});

router.get('/access/download', requireAccessKey, (req, res) => {
  const accessKey = req.accessKey;
  const filePath = req.query.file || '';
  const sanitizedPath = path.normalize(filePath).replace(/^(\.\.(\/|\\|$))+/, '');
  const fullPath = path.join(path.resolve(CONFIG.ACCESS_FOLDER), sanitizedPath);
  if (!fullPath.startsWith(path.resolve(CONFIG.ACCESS_FOLDER))) {
    return res.status(403).json({ error: 'Access denied' });
  }
  try {
    if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
      return res.status(404).json({ error: 'File not found' });
    }
    const fileSize = fs.statSync(fullPath).size;
    keysData.traffic[accessKey] = (keysData.traffic[accessKey] || 0) + fileSize;
    saveKeys();
    res.download(fullPath);
  } catch (err) {
    res.status(500).json({ error: 'Error downloading file' });
  }
});

module.exports = router;
