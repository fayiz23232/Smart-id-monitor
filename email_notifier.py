# email_notifier.py
import smtplib
import ssl
from email.message import EmailMessage
import datetime
import threading # Keep threading here if you want the module to handle async itself (Option B below)
                # Or remove it if the caller (database_manager) handles threading (Option A)

# --- Option A: Synchronous Send Function (Caller handles threading) ---
def send_fine_notification(recipient_email, student_name, fine_amount, total_fine, email_config):
    """Connects to SMTP server and sends a single fine notification email."""
    sender_email = email_config.get('sender_email')
    sender_password = email_config.get('sender_password')
    smtp_server = email_config.get('smtp_server')
    smtp_port = email_config.get('smtp_port')
    use_tls = email_config.get('use_tls', True)
    subject = email_config.get('email_subject', 'Fine Notification')

    if not all([recipient_email, sender_email, sender_password, smtp_server, smtp_port]):
        print(f"  [Email WARN] Missing required email configuration or recipient address for {student_name}. Cannot send.")
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    body = f"""Dear {student_name},

This email is to inform you that a fine of ₹{fine_amount:.2f} has been applied due to an ID card policy violation detected on {datetime.datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}.

Your current total outstanding fine amount is ₹{total_fine:.2f}.

Please ensure you adhere to the ID card policy in the future.

Regards,
System Administration
"""
    msg.set_content(body)

    print(f"  [Email] Attempting to send notification to {student_name} at {recipient_email}...")
    server = None
    try:
        context = ssl.create_default_context()
        if use_tls:
             server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
             server.starttls(context=context)
        else:
             server = smtplib.SMTP_SSL(smtp_server, smtp_port, context=context, timeout=10)

        server.login(sender_email, sender_password)
        server.send_message(msg)
        print(f"  [Email OK] Notification sent successfully to {student_name}.")

    except smtplib.SMTPAuthenticationError:
         print(f"  [Email FAIL] Authentication failed for {sender_email}. Check email/password/app password in config.ini.")
    except smtplib.SMTPServerDisconnected:
         print(f"  [Email FAIL] Server disconnected unexpectedly. Check server/port/network.")
    except smtplib.SMTPConnectError:
         print(f"  [Email FAIL] Could not connect to {smtp_server}:{smtp_port}. Check server/port/firewall.")
    except ConnectionRefusedError:
         print(f"  [Email FAIL] Connection refused by {smtp_server}:{smtp_port}. Check server/port/firewall.")
    except TimeoutError:
         print(f"  [Email FAIL] Connection timed out to {smtp_server}:{smtp_port}.")
    except smtplib.SMTPException as smtp_e:
         print(f"  [Email FAIL] SMTP error sending email to {student_name}: {smtp_e}")
    except OSError as os_e:
         print(f"  [Email FAIL] Network/OS error sending email to {student_name}: {os_e}")
    except Exception as e:
        print(f"  [Email FAIL] An unexpected error occurred sending email to {student_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if server:
                server.quit()
        except Exception:
            pass

# --- Option B: Asynchronous Send Function (Module handles threading) ---
# Uncomment this section and comment out Option A if you prefer this approach
# def send_fine_notification_async(recipient_email, student_name, fine_amount, total_fine, email_config):
#     """Starts a new thread to send the fine notification."""
#     print(f"  [Info] Creating email thread for {student_name}...")
#     email_thread = threading.Thread(
#         target=send_fine_notification, # Target the synchronous function above
#         args=(recipient_email, student_name, fine_amount, total_fine, email_config),
#         daemon=True
#     )
#     email_thread.start()
# ---