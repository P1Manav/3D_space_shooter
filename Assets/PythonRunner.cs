using UnityEngine;
using System.Diagnostics;  // For Process
using System.IO;          // For Path
using UnityEngine.SceneManagement;

public class PythonRunner : MonoBehaviour
{
    private static Process serverProcess; // static so it persists even if scene reloads

    public void StartServerAndLoadLevel1()
    {
        // Kill old server if it's still running
        if (serverProcess != null && !serverProcess.HasExited)
        {
            try
            {
                serverProcess.Kill();
                serverProcess.Dispose();
                UnityEngine.Debug.Log("Old DQN server stopped.");
            }
            catch (System.Exception ex)
            {
                UnityEngine.Debug.LogWarning("Error stopping old server: " + ex.Message);
            }
        }

        string pythonPath = "python"; // Or full path to your python.exe
        string scriptPath = Path.Combine(Application.dataPath, "dqn_server.py");

        ProcessStartInfo startInfo = new ProcessStartInfo();
        startInfo.FileName = pythonPath;
        startInfo.Arguments = "\"" + scriptPath + "\"";
        startInfo.UseShellExecute = false;
        startInfo.RedirectStandardOutput = true;
        startInfo.RedirectStandardError = true;
        startInfo.CreateNoWindow = true;

        try
        {
            serverProcess = Process.Start(startInfo);
            UnityEngine.Debug.Log("New DQN Server started: " + scriptPath);

            // Capture server output
            serverProcess.OutputDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                    UnityEngine.Debug.Log("[SERVER] " + e.Data);
            };
            serverProcess.BeginOutputReadLine();

            // Load Level1 after starting server
            SceneManager.LoadScene("Level1");
        }
        catch (System.Exception ex)
        {
            UnityEngine.Debug.LogError("Failed to start DQN server: " + ex.Message);
        }
    }

    private void OnApplicationQuit()
    {
        // Clean up when quitting the game
        if (serverProcess != null && !serverProcess.HasExited)
        {
            serverProcess.Kill();
            serverProcess.Dispose();
        }
    }
}
