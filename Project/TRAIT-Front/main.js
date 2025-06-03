import { app, BrowserWindow } from 'electron';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import url from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow;
let backendProcess;

const isDev = !app.isPackaged;
const platform = process.platform;

app.on('ready', () => {
    console.log('Starting backend...');

    const exeName = platform === 'win32' ? 'backend.exe' : 'backend';
    const exePath = isDev
        ? path.join(__dirname, 'backend', exeName)
        : path.join(process.resourcesPath, 'backend', exeName);

    backendProcess = spawn(exePath, {
        shell: true,
        detached: false,
        stdio: 'inherit',
        windowsHide: true,
        env: {
            ...process.env,
            ENV: isDev ? 'dev' : 'prod',
        },
    });


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

    mainWindow.on('close', () => {
        console.log('Closing Electron...');
        if (backendProcess) {
            console.log('Sending SIGTERM to backend...');
            backendProcess.kill('SIGTERM');
        }
    });
});

app.on('quit', () => {
    console.log('Electron is quitting...');
    if (backendProcess) {
        console.log('Sending SIGTERM to backend...');
        backendProcess.kill('SIGTERM');
    }
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        console.log('All windows closed, quitting Electron...');
        app.quit();
    }
});