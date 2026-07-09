* Remove sudo password requirements
    ```bash
    sudo visudo -f /etc/sudoers.d/exam
    # Insert a follwoing line
    rubis ALL=(root) NOPASSWD: ALL
    ```