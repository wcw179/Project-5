"""A utility script to securely store secrets in the system keyring."""

import getpass

import keyring

SERVICE_NAME = "BlackSwanHunter"


def set_secret(account: str, secret_name: str):
    """Sets a secret for a given account in the system keyring."""
    try:
        secret_value = getpass.getpass(f"Enter value for '{secret_name}': ")
        keyring.set_password(SERVICE_NAME, f"{account}_{secret_name}", secret_value)
        print(f"Successfully stored '{secret_name}' for account '{account}'.")
    except Exception as e:
        print(f"Failed to store secret: {e}")


def main():
    """Main function to guide user through setting secrets."""
    print(f"--- Storing Secrets for {SERVICE_NAME} ---")
    print("This script will securely store your broker API key and secret.")

    account_name = input("Enter a name for this credential set (e.g., 'mt5_demo'): ")
    if not account_name:
        print("Account name cannot be empty.")
        return

    set_secret(account_name, "api_key")
    set_secret(account_name, "api_secret")

    print("\nSetup complete. Update your config.yml to use this account name.")


if __name__ == "__main__":
    main()
