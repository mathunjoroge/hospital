"""
departments/hl7/mllp_daemon.py
──────────────────────────────
Phase 4 — P4-02 / P4-03: Standalone asyncio MLLP Listener

MLLP (Minimal Lower Layer Protocol) framing:
  Start-Block  : 0x0B
  End-Block    : 0x1C
  Carriage-Ret : 0x0D

This daemon:
  1. Listens on MLLP_HOST:MLLP_PORT (default 0.0.0.0:2576)
  2. Parses incoming ORU^R01 or ADT messages
  3. POSTs to the Flask /api/hl7/oru endpoint over HTTP (never imports Flask directly)
  4. Sends HL7 ACK back to the sending analyser

Run as:
  python -m departments.hl7.mllp_daemon

Environment variables:
  MLLP_HOST          (default: 0.0.0.0)
  MLLP_PORT          (default: 2576)
  HL7_INGEST_URL     (default: http://localhost:5000/api/hl7/oru)
  HL7_INGEST_API_KEY (default: changeme)
  MLLP_LOG_LEVEL     (default: INFO)
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

import aiohttp

# ── MLLP framing constants ────────────────────────────────────────────────────
MLLP_SB = b"\x0b"   # Start Block
MLLP_EB = b"\x1c"   # End Block
MLLP_CR = b"\x0d"   # Carriage Return

# ── Config from environment ───────────────────────────────────────────────────
MLLP_HOST = os.environ.get("MLLP_HOST", "0.0.0.0")  # nosec: B104
MLLP_PORT = int(os.environ.get("MLLP_PORT", "2576"))
HL7_INGEST_URL = os.environ.get("HL7_INGEST_URL", "http://localhost:5000/api/hl7/oru")
HL7_INGEST_API_KEY = os.environ.get("HL7_INGEST_API_KEY", "changeme")
LOG_LEVEL = os.environ.get("MLLP_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [MLLP] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("mllp_daemon")


# ─────────────────────────────────────────────────────────────────────────────
# HL7 ACK builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_ack(raw_msg: str, ack_code: str = "AA", error_msg: str = "") -> bytes:
    """
    Build a minimal HL7v2 ACK message and wrap it in MLLP framing.
    ack_code: AA (accept), AE (error), AR (reject)
    """
    now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    # Extract MSH fields from the incoming message for the ACK header
    lines = raw_msg.replace("\r\n", "\r").split("\r")
    msh = None
    for line in lines:
        if line.startswith("MSH"):
            msh = line.split("|")
            break

    if msh and len(msh) >= 9:
        field_sep = msh[1] if len(msh) > 1 else "^~\\&"
        sending_app = msh[2] if len(msh) > 2 else "HMIS"
        sending_fac = msh[3] if len(msh) > 3 else "KE"
        receiving_app = msh[4] if len(msh) > 4 else "LIS"
        receiving_fac = msh[5] if len(msh) > 5 else "KE"
        msg_ctrl_id = msh[9] if len(msh) > 9 else now
    else:
        field_sep = "^~\\&"
        sending_app = "HMIS"
        sending_fac = "KE"
        receiving_app = "LIS"
        receiving_fac = "KE"
        msg_ctrl_id = now

    ack_ctrl_id = f"ACK{now}"
    err_segment = f"\rERR|{error_msg}" if error_msg else ""

    ack_text = (
        f"MSH|{field_sep}|{sending_app}|{sending_fac}|"
        f"{receiving_app}|{receiving_fac}|{now}||ACK|{ack_ctrl_id}|P|2.5\r"
        f"MSA|{ack_code}|{msg_ctrl_id}{err_segment}\r"
    )

    return MLLP_SB + ack_text.encode("utf-8") + MLLP_EB + MLLP_CR


# ─────────────────────────────────────────────────────────────────────────────
# HTTP forwarder → Flask /api/hl7/oru
# ─────────────────────────────────────────────────────────────────────────────

async def _forward_to_flask(session: aiohttp.ClientSession, raw_hl7: str) -> bool:
    """
    POST raw HL7 text to Flask ingest endpoint.
    Returns True on HTTP 201, False otherwise.
    """
    headers = {
        "Content-Type": "x-application/hl7-v2+er7",
        "X-HL7-API-Key": HL7_INGEST_API_KEY,
    }
    try:
        async with session.post(
            HL7_INGEST_URL,
            data=raw_hl7.encode("utf-8"),
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            body = await resp.text()
            if resp.status == 201:
                logger.debug("Flask ingest OK: %s", body[:120])
                return True
            logger.warning("Flask ingest HTTP %s: %s", resp.status, body[:200])
            return False
    except aiohttp.ClientError as exc:
        logger.error("Flask ingest connection error: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MLLP connection handler
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    http_session: aiohttp.ClientSession,
) -> None:
    peer = writer.get_extra_info("peername")
    logger.info("MLLP connection from %s", peer)
    buffer = b""

    try:
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            buffer += chunk

            # Process all complete MLLP frames in buffer
            while MLLP_SB in buffer:
                sb_idx = buffer.index(MLLP_SB)
                eb_seq = MLLP_EB + MLLP_CR
                if eb_seq not in buffer[sb_idx:]:
                    break  # incomplete frame — wait for more data

                eb_idx = buffer.index(eb_seq, sb_idx)
                raw_bytes = buffer[sb_idx + 1 : eb_idx]
                buffer = buffer[eb_idx + len(eb_seq):]

                raw_msg = raw_bytes.decode("utf-8", errors="replace")
                msg_type = "UNKNOWN"
                for line in raw_msg.replace("\r\n", "\r").split("\r"):
                    if line.startswith("MSH"):
                        parts = line.split("|")
                        if len(parts) >= 9:
                            msg_type = parts[8].replace("^", "_")
                        break

                logger.info("Received %s (%d bytes) from %s", msg_type, len(raw_bytes), peer)

                if "ORU" in msg_type or "ADT" in msg_type:
                    success = await _forward_to_flask(http_session, raw_msg)
                    ack_code = "AA" if success else "AE"
                    err = "" if success else "Flask ingest error"
                else:
                    logger.debug("Ignoring unsupported message type %s", msg_type)
                    ack_code = "AR"
                    err = f"Unsupported message type: {msg_type}"

                ack = _build_ack(raw_msg, ack_code, err)
                writer.write(ack)
                await writer.drain()
                logger.debug("Sent ACK %s to %s", ack_code, peer)

    except asyncio.IncompleteReadError:
        logger.debug("Connection closed by %s", peer)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error handling connection from %s: %s", peer, exc)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # pylint: disable=broad-except
            pass
        logger.info("MLLP connection closed: %s", peer)


# ─────────────────────────────────────────────────────────────────────────────
# Main server loop
# ─────────────────────────────────────────────────────────────────────────────

async def _main() -> None:
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as http_session:

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            await _handle_connection(reader, writer, http_session)

        server = await asyncio.start_server(handle, MLLP_HOST, MLLP_PORT)
        addr = server.sockets[0].getsockname()
        logger.info("MLLP daemon listening on %s:%s  →  %s", addr[0], addr[1], HL7_INGEST_URL)

        async with server:
            await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("MLLP daemon stopped")
