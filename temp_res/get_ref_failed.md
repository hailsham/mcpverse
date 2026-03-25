# get_ref 失败的 10 道题

## 失败原因分类

### MCP: mcp-paperswithcode (6题) — Connection failed to MCP server
| question_id | get_answer_id | 问题 |
|---|---|---|
| Q261 | q21 | Based on info of paper with code, what is the title of Kaiming He's last paper posted on arXiv in 2025? |
| Q262 | q22 | According to the information on Paper with Code, whose latest published paper has a more recent release date: Kaiming He or Numaan Naeem? |
| Q263 | q23 | What is the tasks included in He Kaiming's latest paper according to the information on Paper with Code? |
| Q288 | q48 | What is the tasks included in Ross Girshick's latest paper according to the information on Paper with Code? |
| Q300 | q60 | Based on info of paper with code, what is the title of Xiaogang Wang's last paper posted on arXiv? |
| Q301 | q61 | According to the information on Paper with Code, whose latest published paper has a more recent release date: Chenyuan Wu or Numaan Naeem? |
| Q302 | q62 | What is the tasks included in Chenyuan Wu's latest paper according to the information on Paper with Code? |

### MCP: weibo (2题) — API 返回 432 / 数据结构异常
| question_id | get_answer_id | 问题 |
|---|---|---|
| Q289 | q49 | What is the id of the first result of searching users with keyword "Hansen Yang" in Weibo? |
| Q290 | q50 | What is the id of the first result of searching users with keyword 'apple' in Weibo? |

### MCP: geeknews-mcp-server (1题) — Connection failed to MCP server
| question_id | get_answer_id | 问题 |
|---|---|---|
| Q248 | q7 | Retrieve the url of top 1 article from GeekNews. |

## 根本原因
- **mcp-paperswithcode**: MCP server 连接失败，`error: unknown option '--key'`，smithery 的 npx 启动参数有问题
- **weibo**: API 返回 432 (需要登录态) 或数据结构变更
- **geeknews**: MCP server 连接超时
