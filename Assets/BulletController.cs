using UnityEngine;

public class BulletController : MonoBehaviour
{
    public float speed = 100f;

    void Update()
    {
        transform.position += transform.forward * speed * Time.deltaTime;
    }

    void OnTriggerEnter(Collider other)
    {
        Destroy(gameObject);
    }
}
