{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
{%- if fullname in noindex %}
   :no-index:
{%- endif %}
