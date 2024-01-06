---
layout: page-no-header
title: Notes
permalink: /notes/
---

### Reinforcment Learning:
{% for chapter in site.data.RL_content %}
- [{{ chapter.title }}]({{ chapter.path }}): {{ chapter.description }}
{% endfor %}
