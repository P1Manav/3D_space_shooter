using UnityEngine;

public class DispatcherBootstrap : MonoBehaviour
{
    void Awake()
    {
        UnityMainThreadDispatcher.Instance();  // Ensure it's created early
    }
}
