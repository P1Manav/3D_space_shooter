using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class MenuManager : MonoBehaviour
{
    [Header("Panels")]
    public GameObject mainMenuPanel;      // assign MainMenuPanel
    public GameObject levelSelectPanel;   // assign LevelSelectPanel
    public GameObject comingSoonPanel;    // assign ComingSoonPanel (optional)

    [Header("Buttons")]
    public Button level2Button;           // assign Level2Button (optional)

    void Start()
    {
        // initial UI state
        ShowMainMenu();

        if (comingSoonPanel != null) comingSoonPanel.SetActive(false);

        // disable Level 2 since it's in development
        if (level2Button != null)
        {
            level2Button.interactable = false; // visually disabled
        }
    }

    // Panel toggles
    public void ShowMainMenu()
    {
        if (mainMenuPanel != null) mainMenuPanel.SetActive(true);
        if (levelSelectPanel != null) levelSelectPanel.SetActive(false);
    }

    public void ShowLevelSelect()
    {
        if (mainMenuPanel != null) mainMenuPanel.SetActive(false);
        if (levelSelectPanel != null) levelSelectPanel.SetActive(true);
    }

    // Load scenes by name (make sure names match your saved scenes)
    public void PlayLevel1()
    {
        SceneManager.LoadScene("Level1");
    }

    // Called if user clicks Level2. Since Level2 is in dev, show Coming Soon.
    public void PlayLevel2()
    {
        if (comingSoonPanel != null)
        {
            comingSoonPanel.SetActive(true);
            return;
        }
        // fallback if you later enable Level2
        SceneManager.LoadScene("Level2");
    }

    // Close the coming soon popup
    public void CloseComingSoon()
    {
        if (comingSoonPanel != null) comingSoonPanel.SetActive(false);
    }

    // If you want a button in Level2 scene to go back:
    public void BackToMenu()
    {
        SceneManager.LoadScene("MainMenu");
    }

    public void QuitGame()
    {
        Debug.Log("Quit Game");
    #if UNITY_EDITOR
        EditorApplication.isPlaying = false;
    #else
        Application.Quit();
    #endif
    }
}
