import logging
from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_wtf.csrf import CSRFError

logger = logging.getLogger(__name__)


def is_api_request():
    """Check if the request originates from an API endpoint, JSON request, or AJAX client."""
    return (
        request.path.startswith("/api/")
        or request.is_json
        or request.headers.get("Accept") == "application/json"
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
    )


def register_error_handlers(app, login_manager=None):
    """
    Register application-wide custom error handlers for CSRF validation,
    authentication, authorization, and standard HTTP error codes.
    """

    # 1. Custom Flask-Login Unauthorized Handler (Unauthenticated User Access)
    if login_manager:
        @login_manager.unauthorized_handler
        def custom_unauthorized():
            if is_api_request():
                return jsonify({
                    "error": "Authentication Required",
                    "message": "You must be signed in with a valid session or authentication token to access this clinical module.",
                    "code": 401,
                }), 401
            flash(
                "Authentication Required: Please sign in to your account to access this section of the system.",
                "warning",
            )
            return redirect(url_for("login", next=request.url))

    # 2. CSRF Token Missing or Invalid Error Handler
    @app.errorhandler(CSRFError)
    def handle_csrf_error(e):
        reason = getattr(e, "description", "CSRF token missing or invalid")
        logger.warning(f"CSRF validation failure on {request.path}: {reason}")
        if is_api_request():
            return jsonify({
                "error": "Security Verification Failed",
                "message": "Security verification failed (CSRF token missing or expired). Please refresh the page and try again.",
                "code": 400,
            }), 400
        flash(
            "Security session expired or form token invalid. Please refresh the page and try again.",
            "error",
        )
        return render_template(
            "errors/error.html",
            error_code=400,
            error_title="Security Verification Failed",
            error_message="Your form security token was missing or has expired. Please refresh the page and resubmit your request.",
        ), 400

    # 3. HTTP 401 Unauthorized Handler
    @app.errorhandler(401)
    def handle_401(e):
        description = getattr(e, "description", None)
        msg = description if (description and description != "Unauthorized" and len(description) > 5) else "You must be signed in with a valid session or token to perform this operation."
        if is_api_request():
            return jsonify({
                "error": "Authentication Required",
                "message": msg,
                "code": 401,
            }), 401
        flash("Authentication Required: Your session has expired or login is required.", "warning")
        return redirect(url_for("login", next=request.url))

    # 4. HTTP 403 Forbidden Handler
    @app.errorhandler(403)
    def handle_403(e):
        description = getattr(e, "description", None)
        msg = description if (description and description != "Forbidden" and len(description) > 5) else "Your account role does not have permission to access this clinical module or feature."
        if is_api_request():
            return jsonify({
                "error": "Access Denied",
                "message": msg,
                "code": 403,
            }), 403
        return render_template(
            "errors/error.html",
            error_code=403,
            error_title="Access Denied",
            error_message=msg,
        ), 403

    # 5. HTTP 404 Not Found Handler
    @app.errorhandler(404)
    def handle_404(e):
        if is_api_request():
            return jsonify({
                "error": "Resource Not Found",
                "message": "The requested clinical record, page, or API endpoint could not be found.",
                "code": 404,
            }), 404
        return render_template(
            "errors/error.html",
            error_code=404,
            error_title="Resource Not Found",
            error_message="The requested page or record could not be found. Please check the URL or navigate back to the home page.",
        ), 404

    # 6. HTTP 400 Bad Request Handler
    @app.errorhandler(400)
    def handle_400(e):
        description = getattr(e, "description", None)
        msg = description if (description and description != "Bad Request" and len(description) > 5) else "The request contained missing or invalid data."
        if is_api_request():
            return jsonify({
                "error": "Invalid Request",
                "message": msg,
                "code": 400,
            }), 400
        flash(f"Invalid Request: {msg}", "error")
        return render_template(
            "errors/error.html",
            error_code=400,
            error_title="Invalid Request",
            error_message=msg,
        ), 400

    # 7. HTTP 429 Rate Limit Exceeded Handler
    @app.errorhandler(429)
    def handle_429(e):
        msg = "You have submitted too many requests in a short period. Please wait a moment before trying again."
        if is_api_request():
            return jsonify({
                "error": "Too Many Requests",
                "message": msg,
                "code": 429,
            }), 429
        flash(msg, "warning")
        return render_template(
            "errors/error.html",
            error_code=429,
            error_title="Rate Limit Exceeded",
            error_message=msg,
        ), 429

    # 8. HTTP 500 Internal Server Error Handler
    @app.errorhandler(500)
    def handle_500(e):
        logger.exception("Internal server error occurred:")
        msg = "An unexpected error occurred while processing your request. The technical team has been notified."
        if is_api_request():
            return jsonify({
                "error": "Internal System Error",
                "message": msg,
                "code": 500,
            }), 500
        return render_template(
            "errors/error.html",
            error_code=500,
            error_title="Internal System Error",
            error_message=msg,
        ), 500
