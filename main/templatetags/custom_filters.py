from django import template
import base64

register = template.Library()

@register.filter(name='base64')
def base64_filter(value):
    """Convert image to base64 encoding."""
    with open(value, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
