import { app, BrowserWindow } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import url from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow;
const isDev = !app.isPackaged;

app.on('ready', () => {
    console.log('Launching Electron window...');

    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
        },
    });

    const indexPath = isDev
        ? 'http://localhost:5173'
        : url.format({
            pathname: path.join(__dirname, 'dist', 'index.html'),
            protocol: 'file:',
            slashes: true,
        });

    mainWindow.loadURL(indexPath);

    mainWindow.on('closed', () => {
        console.log('Electron window closed');
        mainWindow = null;
    });
});

app.on('quit', () => {
    console.log('Electron is quitting...');
    app.quit();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        console.log('All windows closed, quitting Electron...');
        app.quit();
    }
});