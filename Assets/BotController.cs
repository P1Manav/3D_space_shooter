using UnityEngine;

public class BotController : MonoBehaviour
{
    public Transform firePoint;
    public GameObject bulletPrefab; // must be a prefab from Project window
    public string agentId = "bot_1";
    public string ownerTag = "Bot";
    public float moveSpeed = 12f;
    public float rotationMultiplier = 1f;
    public float fov = 25f;
    public float bulletSpeed = 50f;

    private float yawDelta, pitchDelta, rollDelta;
    private bool shootRequested;
    private Collider myCollider;

    void Awake()
    {
        myCollider = GetComponent<Collider>();
    }

    public void SetPrediction(float yaw, float pitch, float roll, bool shoot)
    {
        yawDelta = yaw;
        pitchDelta = pitch;
        rollDelta = roll;
        shootRequested = shoot;
    }

    void Update()
    {
        transform.Translate(Vector3.forward * moveSpeed * Time.deltaTime);
        Vector3 rotChange = new Vector3(pitchDelta, yawDelta, rollDelta) * rotationMultiplier * Time.deltaTime * 60f;
        transform.Rotate(rotChange, Space.Self);

        if (shootRequested)
        {
            GameObject player = GameObject.FindWithTag("Player");
            if (player != null)
            {
                Vector3 toPlayer = (player.transform.position - transform.position).normalized;
                float angle = Vector3.Angle(transform.forward, toPlayer);
                if (angle <= fov)
                {
                    MainThreadDispatcher.Enqueue(() => Fire());
                }
            }
            shootRequested = false;
        }
    }

    void Fire()
    {
        if (!bulletPrefab || !firePoint) return;

        // Create a new clone so the original prefab is never touched
        GameObject bulletClone = Instantiate(bulletPrefab, firePoint.position, firePoint.rotation);

        Bullet bulletScript = bulletClone.GetComponent<Bullet>();
        if (bulletScript != null)
        {
            bulletScript.ownerTag = ownerTag;
            bulletScript.shooterId = agentId;
        }

        Rigidbody rbBullet = bulletClone.GetComponent<Rigidbody>();
        if (rbBullet != null) rbBullet.linearVelocity = firePoint.forward * bulletSpeed;

        Collider bulletCol = bulletClone.GetComponent<Collider>();
        if (bulletCol && myCollider) Physics.IgnoreCollision(bulletCol, myCollider);

        // Destroy the clone after 8 seconds (prefab stays intact)
        Destroy(bulletClone, 8f);
    }
}
