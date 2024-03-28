const { app, BrowserWindow } = require('electron');
console.log('hhh')
function createWindow() {
  console.log('hhh2')
  // 创建浏览器窗口
  let win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true
    }
  });

  // 加载React应用
  win.loadURL('http://localhost:3000');
}

app.whenReady().then(createWindow);
