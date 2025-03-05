using UnityEngine;

public class bullet : MonoBehaviour
{
    public float speed = 50f;
    public float lifetime = 5f;

    void Start()
    {
        GetComponent<Rigidbody>().linearVelocity = transform.forward * speed;
        Destroy(gameObject, lifetime); // Destroy bullet after 5 seconds
    }

    void OnCollisionEnter(Collision other)
    {
        // Destroy bullet on impact
        Destroy(gameObject);
    }
}
