---
layout: page-no-header
title: Notes
permalink: /notes/
---

## This is the place where I store all the notes!

### Reinforcment Learning:
{% for chapter in site.data.RL_content %}
- [{{ chapter.title }}]({{ chapter.path }}): {{ chapter.description }}
{% endfor %}
