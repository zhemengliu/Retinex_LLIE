import os
import smtplib
import ssl
from email.message import EmailMessage
import mimetypes
from typing import Optional, Dict, Any, Sequence


def _get_bool(val: Optional[str], default: bool = True) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def send_email_report(
    subject: str,
    body_html: str,
    *,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    sender: Optional[str] = None,
    recipients: Optional[Sequence[str]] = None,
    use_tls: Optional[bool] = None,
    attachments: Optional[Sequence[str]] = None,
) -> bool:
    """Send an HTML email. SMTP configuration can come from args or ENV.

    Environment fallbacks:
      EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO (comma-separated), EMAIL_USE_TLS
    """
    try:
        host = smtp_host or os.getenv("EMAIL_HOST")
        port = int(smtp_port or os.getenv("EMAIL_PORT") or 465)
        user = username or os.getenv("EMAIL_USER")
        pwd = password or os.getenv("EMAIL_PASS")
        frm = sender or os.getenv("EMAIL_FROM") or user
        to_list = list(recipients) if recipients else [x.strip() for x in (os.getenv("EMAIL_TO") or "").split(",") if x.strip()]
        tls = _get_bool(os.getenv("EMAIL_USE_TLS"), True) if use_tls is None else bool(use_tls)

        if not host or not frm or not to_list:
            print("[Email] Missing required configuration: host/from/to. Skipping send.")
            return False

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = frm
        msg["To"] = ", ".join(to_list)
        # Fallback plain text
        msg.set_content("This email requires an HTML-capable client.")
        msg.add_alternative(body_html, subtype="html")

        # Attach files if provided
        if attachments:
            for path in attachments:
                try:
                    if not path or not os.path.isfile(path):
                        continue
                    ctype, encoding = mimetypes.guess_type(path)
                    if ctype is None or encoding is not None:
                        ctype = 'application/octet-stream'
                    maintype, subtype = ctype.split('/', 1)
                    with open(path, 'rb') as f:
                        data = f.read()
                    filename = os.path.basename(path)
                    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
                except Exception as e:
                    print(f"[Email] Attach failed for {path}: {e}")

        context = ssl.create_default_context()
        # Prefer SMTPS (465). If TLS requested and port is not 465, try STARTTLS.
        if port == 465 and tls:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as server:
                if user and pwd:
                    server.login(user, pwd)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.ehlo()
                if tls:
                    server.starttls(context=context)
                    server.ehlo()
                if user and pwd:
                    server.login(user, pwd)
                server.send_message(msg)
        print(f"[Email] Sent to {to_list} with subject: {subject}")
        return True
    except Exception as e:
        print(f"[Email] Failed to send: {e}")
        return False


def _fmt_val(v: Any) -> str:
    try:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    except Exception:
        return str(v)


def build_training_summary_html(
    title: str,
    best_epoch: Optional[int],
    best_metrics: Optional[Dict[str, Any]],
    best_loss: Optional[float],
    total_epochs: Optional[int],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    metrics_rows = ""
    if best_metrics:
        for k in ["ssim", "psnr", "lpips", "score"]:
            if k in best_metrics:
                metrics_rows += f"<tr><td>{k.upper()}</td><td>{_fmt_val(best_metrics[k])}</td></tr>"
    extra_rows = ""
    if extra:
        for k, v in extra.items():
            extra_rows += f"<tr><td>{k}</td><td>{_fmt_val(v)}</td></tr>"

    return f"""
    <html>
      <body>
        <h2>{title}</h2>
        <table border="1" cellpadding="6" cellspacing="0">
          <tr><th align="left">Field</th><th align="left">Value</th></tr>
          <tr><td>Best Epoch</td><td>{_fmt_val(best_epoch) if best_epoch is not None else '-'}</td></tr>
          <tr><td>Total Epochs</td><td>{_fmt_val(total_epochs) if total_epochs is not None else '-'}</td></tr>
          <tr><td>Best Train Loss</td><td>{_fmt_val(best_loss) if best_loss is not None else '-'}</td></tr>
          {metrics_rows}
          {extra_rows}
        </table>
        <p>Auto-generated notification from LLIE training.</p>
      </body>
    </html>
    """.strip()


def send_best_metrics_email(
    args: Optional[object],
    *,
    title: str,
    best_epoch: Optional[int],
    best_metrics: Optional[Dict[str, Any]],
    best_loss: Optional[float],
    total_epochs: Optional[int],
    attachments: Optional[Sequence[str]] = None,
) -> bool:
    """High-level helper: read SMTP config from args or ENV and send the summary."""
    smtp_host = getattr(args, "email_host", None) if args is not None else None
    smtp_port = getattr(args, "email_port", None) if args is not None else None
    username = getattr(args, "email_user", None) if args is not None else None
    password = getattr(args, "email_pass", None) if args is not None else None
    sender = getattr(args, "email_from", None) if args is not None else None
    recipients = None
    if args is not None and getattr(args, "email_to", None):
        # support comma-separated in args
        if isinstance(args.email_to, (list, tuple)):
            recipients = list(args.email_to)
        else:
            recipients = [x.strip() for x in str(args.email_to).split(',') if x.strip()]
    use_tls = getattr(args, "email_use_tls", None) if args is not None else None

    html = build_training_summary_html(title, best_epoch, best_metrics, best_loss, total_epochs, extra=None)
    subject = f"LLIE Training: {title}"
    return send_email_report(
        subject,
        html,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        username=username,
        password=password,
        sender=sender,
        recipients=recipients,
        use_tls=use_tls,
        attachments=attachments,
    )

