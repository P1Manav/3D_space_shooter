using UnityEngine;

public class PlanetCollider : MonoBehaviour
{
    void OnCollisionEnter(Collision collision)
    {
        // If Player or Bot hits the planet → destroy them
        if (collision.gameObject.CompareTag("Player") || collision.gameObject.CompareTag("Bot"))
        {
            Destroy(collision.gameObject);
        }
    }

    void OnTriggerEnter(Collider other)
    {
        // Also handle trigger collisions if planet collider is set to trigger
        if (other.CompareTag("Player") || other.CompareTag("Bot"))
        {
            Destroy(other.gameObject);
        }
    }
}
