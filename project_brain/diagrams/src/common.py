"""Shared setup: newline-safe label escaping for the emitter.
A literal LF inside an XML attribute is normalised to a space by every XML parser
(draw.io included), so multi-line labels must be written as &#xa;."""
from xml.sax.saxutils import escape as _esc
import bpmn_emit as E
E.escape = lambda s, *a: _esc(s, {"\n": "&#xa;", '"': "&quot;"})
DRV, CT, POL, INS, SHOP = "#eaf1f7", "#eef4ec", "#f3eef7", "#fdf3e7", "#f5f2ea"
