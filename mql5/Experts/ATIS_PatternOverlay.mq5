//+------------------------------------------------------------------+
//|                                          ATIS_PatternOverlay.mq5 |
//| Real-time Explainable-AI pattern drawings for ATIS live trading  |
//| Reads: Common\Files\ATIS\overlay_state.json  (FILE_COMMON)       |
//|     or MQL5\Files\ATIS\overlay_state.json                        |
//+------------------------------------------------------------------+
#property copyright "ATIS"
#property version   "1.00"
#property strict

input int    InpPollMs          = 250;     // Poll interval (ms)
input bool   InpShowLegend      = true;    // Draw legend panel
input bool   InpUseCommonFiles  = true;    // Prefer Common\Files
input string InpStateFile       = "ATIS\\overlay_state.json";
input string InpPrefix          = "ATIS_";  // Object name prefix
input bool   InpClearOnDeinit   = false;   // Remove drawings on remove
input int    InpMaxObjects      = 500;     // Safety cap

int      g_timer_ms = 250;
long     g_last_seq = -1;
string   g_drawn[];

//+------------------------------------------------------------------+
int OnInit()
  {
   g_timer_ms = MathMax(100, InpPollMs);
   EventSetMillisecondTimer(g_timer_ms);
   Print("ATIS Pattern Overlay started. File=", InpStateFile,
         " common=", InpUseCommonFiles);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(InpClearOnDeinit)
      DeleteAllAtis();
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   ProcessOverlayFile();
  }

//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   // Click: show Comment with object tooltip / description
   if(id == CHARTEVENT_OBJECT_CLICK)
     {
      if(StringFind(sparam, InpPrefix) == 0)
        {
         string tip = ObjectGetString(0, sparam, OBJPROP_TOOLTIP);
         if(tip == "" || tip == NULL)
            tip = ObjectGetString(0, sparam, OBJPROP_TEXT);
         if(tip != "" && tip != NULL)
           {
            Comment("══ ATIS Pattern ══\n", tip);
            ChartRedraw(0);
           }
        }
     }
  }

//+------------------------------------------------------------------+
color HexToColor(string hex)
  {
   // Expect RRGGBB
   string h = hex;
   StringReplace(h, "#", "");
   if(StringLen(h) < 6)
      return clrDodgerBlue;
   int r = (int)StringToInteger("0x" + StringSubstr(h, 0, 2));
   int g = (int)StringToInteger("0x" + StringSubstr(h, 2, 2));
   int b = (int)StringToInteger("0x" + StringSubstr(h, 4, 2));
   return (color)((b << 16) | (g << 8) | r); // MT5 COLOR = BGR
  }

//+------------------------------------------------------------------+
bool FileReadAll(const string path, const bool use_common, string &out)
  {
   int flags = FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ;
   if(use_common)
      flags |= FILE_COMMON;
   int h = FileOpen(path, flags);
   if(h == INVALID_HANDLE)
      return false;
   out = "";
   while(!FileIsEnding(h))
     {
      string line = FileReadString(h);
      out += line;
      if(!FileIsEnding(h))
         out += "\n";
     }
   FileClose(h);
   return (StringLen(out) > 0);
  }

//+------------------------------------------------------------------+
string JsonExtractString(const string json, const string key)
  {
   // Minimal extractor: "key"\s*:\s*"value"
   string pat = "\"" + key + "\"";
   int p = StringFind(json, pat);
   if(p < 0)
      return "";
   int colon = StringFind(json, ":", p + StringLen(pat));
   if(colon < 0)
      return "";
   int q1 = StringFind(json, "\"", colon + 1);
   if(q1 < 0)
      return "";
   int q2 = q1 + 1;
   while(q2 < StringLen(json))
     {
      ushort ch = StringGetCharacter(json, q2);
      if(ch == '"' && StringGetCharacter(json, q2 - 1) != '\\')
         break;
      q2++;
     }
   return StringSubstr(json, q1 + 1, q2 - q1 - 1);
  }

//+------------------------------------------------------------------+
long JsonExtractLong(const string json, const string key)
  {
   string pat = "\"" + key + "\"";
   int p = StringFind(json, pat);
   if(p < 0)
      return 0;
   int colon = StringFind(json, ":", p + StringLen(pat));
   if(colon < 0)
      return 0;
   int i = colon + 1;
   while(i < StringLen(json) && (StringGetCharacter(json, i) == ' ' || StringGetCharacter(json, i) == '\t'))
      i++;
   string num = "";
   while(i < StringLen(json))
     {
      ushort ch = StringGetCharacter(json, i);
      if((ch >= '0' && ch <= '9') || ch == '-' )
         num += CharToString((uchar)ch);
      else
         break;
      i++;
     }
   return (long)StringToInteger(num);
  }

//+------------------------------------------------------------------+
double JsonExtractDouble(const string json, const string key)
  {
   string pat = "\"" + key + "\"";
   int p = StringFind(json, pat);
   if(p < 0)
      return 0.0;
   int colon = StringFind(json, ":", p + StringLen(pat));
   if(colon < 0)
      return 0.0;
   int i = colon + 1;
   while(i < StringLen(json) && (StringGetCharacter(json, i) == ' ' || StringGetCharacter(json, i) == '\t'))
      i++;
   string num = "";
   while(i < StringLen(json))
     {
      ushort ch = StringGetCharacter(json, i);
      if((ch >= '0' && ch <= '9') || ch == '-' || ch == '.' || ch == 'e' || ch == 'E' || ch == '+')
         num += CharToString((uchar)ch);
      else
         break;
      i++;
     }
   return StringToDouble(num);
  }

//+------------------------------------------------------------------+
bool JsonExtractBool(const string json, const string key)
  {
   string pat = "\"" + key + "\"";
   int p = StringFind(json, pat);
   if(p < 0)
      return false;
   int colon = StringFind(json, ":", p + StringLen(pat));
   if(colon < 0)
      return false;
   if(StringFind(json, "true", colon + 1) == colon + 1 || StringFind(json, "true", colon + 1) == colon + 2)
      return true;
   // tolerate whitespace
   int i = colon + 1;
   while(i < StringLen(json) && StringGetCharacter(json, i) <= ' ')
      i++;
   return (StringFind(json, "true", i) == i);
  }

//+------------------------------------------------------------------+
void DeleteAllAtis()
  {
   int total = ObjectsTotal(0, -1, -1);
   for(int i = total - 1; i >= 0; i--)
     {
      string name = ObjectName(0, i, -1, -1);
      if(StringFind(name, InpPrefix) == 0)
         ObjectDelete(0, name);
     }
   ArrayResize(g_drawn, 0);
  }

//+------------------------------------------------------------------+
void TrackDrawn(const string name)
  {
   int n = ArraySize(g_drawn);
   for(int i = 0; i < n; i++)
      if(g_drawn[i] == name)
         return;
   ArrayResize(g_drawn, n + 1);
   g_drawn[n] = name;
  }

//+------------------------------------------------------------------+
void PruneMissing(string &keep[])
  {
   int n = ArraySize(g_drawn);
   for(int i = n - 1; i >= 0; i--)
     {
      bool found = false;
      for(int j = 0; j < ArraySize(keep); j++)
        {
         if(g_drawn[i] == keep[j])
           {
            found = true;
            break;
           }
        }
      if(!found)
        {
         ObjectDelete(0, g_drawn[i]);
        }
     }
   ArrayResize(g_drawn, 0);
   for(int k = 0; k < ArraySize(keep); k++)
     {
      TrackDrawn(keep[k]);
     }
  }

//+------------------------------------------------------------------+
void EnsureArrow(const string name, datetime t, double price, color clr, int code, int width, const string tip)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_ARROW, 0, t, price);
   ObjectSetInteger(0, name, OBJPROP_TIME, t);
   ObjectSetDouble(0, name, OBJPROP_PRICE, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE, code);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, true);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, false);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);
  }

//+------------------------------------------------------------------+
void EnsureText(const string name, datetime t, double price, color clr, const string text, int fontsize, const string tip)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_TEXT, 0, t, price);
   ObjectSetInteger(0, name, OBJPROP_TIME, t);
   ObjectSetDouble(0, name, OBJPROP_PRICE, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontsize);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, true);
  }

//+------------------------------------------------------------------+
void EnsureTrend(const string name, datetime t1, double p1, datetime t2, double p2, color clr, int width, int style, const string tip)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, t2, p2);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t1);
   ObjectSetDouble(0, name, OBJPROP_PRICE, 0, p1);
   ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
   ObjectSetDouble(0, name, OBJPROP_PRICE, 1, p2);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, true);
  }

//+------------------------------------------------------------------+
void EnsureRect(const string name, datetime t1, double p1, datetime t2, double p2, color clr, int width, bool fill, const string tip)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t1);
   ObjectSetDouble(0, name, OBJPROP_PRICE, 0, p1);
   ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
   ObjectSetDouble(0, name, OBJPROP_PRICE, 1, p2);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, name, OBJPROP_FILL, fill);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, true);
  }

//+------------------------------------------------------------------+
void DrawLegend(const string json)
  {
   if(!InpShowLegend)
      return;
   // Fixed corner legend via OBJ_LABEL
   string title = InpPrefix + "LEGEND_TITLE";
   if(ObjectFind(0, title) < 0)
      ObjectCreate(0, title, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, title, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, title, OBJPROP_XDISTANCE, 12);
   ObjectSetInteger(0, title, OBJPROP_YDISTANCE, 18);
   ObjectSetString(0, title, OBJPROP_TEXT, "ATIS Pattern Legend");
   ObjectSetInteger(0, title, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, title, OBJPROP_FONTSIZE, 9);
   TrackDrawn(title);

   // Parse legend array entries loosely (up to 6)
   string legend_keys[6] = {"Candlestick", "Chart / Market", "Compound", "BOS / CHOCH", "Trade-Linked", "Invalidated"};
   string legend_colors[6] = {"26A69A", "29B6F6", "AB47BC", "FFD54F", "FFFFFF", "616161"};
   // Prefer colors from payload legend if present
   int pos = StringFind(json, "\"legend\"");
   if(pos >= 0)
     {
      // keep defaults; labels are stable
     }
   for(int i = 0; i < 6; i++)
     {
      string nm = InpPrefix + "LEGEND_" + IntegerToString(i);
      if(ObjectFind(0, nm) < 0)
         ObjectCreate(0, nm, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, nm, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, nm, OBJPROP_XDISTANCE, 12);
      ObjectSetInteger(0, nm, OBJPROP_YDISTANCE, 36 + i * 14);
      ObjectSetString(0, nm, OBJPROP_TEXT, "■ " + legend_keys[i]);
      ObjectSetInteger(0, nm, OBJPROP_COLOR, HexToColor(legend_colors[i]));
      ObjectSetInteger(0, nm, OBJPROP_FONTSIZE, 8);
      TrackDrawn(nm);
     }
  }

//+------------------------------------------------------------------+
int FindMatchingBrace(const string s, const int open_idx, const ushort open_ch, const ushort close_ch)
  {
   int depth = 0;
   bool in_str = false;
   for(int i = open_idx; i < StringLen(s); i++)
     {
      ushort ch = StringGetCharacter(s, i);
      if(ch == '"' && (i == 0 || StringGetCharacter(s, i - 1) != '\\'))
         in_str = !in_str;
      if(in_str)
         continue;
      if(ch == open_ch)
         depth++;
      else if(ch == close_ch)
        {
         depth--;
         if(depth == 0)
            return i;
        }
     }
   return -1;
  }

//+------------------------------------------------------------------+
void ProcessOneObject(const string obj_json, string &keep[])
  {
   string type = JsonExtractString(obj_json, "type");
   string name = JsonExtractString(obj_json, "name");
   if(name == "" || StringFind(name, InpPrefix) != 0)
      return;
   if(ArraySize(keep) >= InpMaxObjects)
      return;
   color clr = HexToColor(JsonExtractString(obj_json, "color"));
   string tip = JsonExtractString(obj_json, "tooltip");
   StringReplace(tip, "\\n", "\n");
   int width = (int)JsonExtractLong(obj_json, "width");
   if(width <= 0)
      width = 1;
   int style = (int)JsonExtractLong(obj_json, "style");

   if(type == "arrow")
     {
      datetime t = (datetime)JsonExtractLong(obj_json, "time");
      double price = JsonExtractDouble(obj_json, "price");
      int code = (int)JsonExtractLong(obj_json, "arrow_code");
      if(code <= 0)
         code = 159;
      EnsureArrow(name, t, price, clr, code, width, tip);
     }
   else if(type == "text")
     {
      datetime t = (datetime)JsonExtractLong(obj_json, "time");
      double price = JsonExtractDouble(obj_json, "price");
      string text = JsonExtractString(obj_json, "text");
      StringReplace(text, "\\n", "\n");
      int fs = (int)JsonExtractLong(obj_json, "fontsize");
      if(fs <= 0)
         fs = 8;
      EnsureText(name, t, price, clr, text, fs, tip);
     }
   else if(type == "trendline")
     {
      datetime t1 = (datetime)JsonExtractLong(obj_json, "t1");
      datetime t2 = (datetime)JsonExtractLong(obj_json, "t2");
      double p1 = JsonExtractDouble(obj_json, "p1");
      double p2 = JsonExtractDouble(obj_json, "p2");
      EnsureTrend(name, t1, p1, t2, p2, clr, width, style, tip);
     }
   else if(type == "rectangle")
     {
      datetime t1 = (datetime)JsonExtractLong(obj_json, "t1");
      datetime t2 = (datetime)JsonExtractLong(obj_json, "t2");
      double p1 = JsonExtractDouble(obj_json, "p1");
      double p2 = JsonExtractDouble(obj_json, "p2");
      bool fill = JsonExtractBool(obj_json, "fill");
      EnsureRect(name, t1, p1, t2, p2, clr, width, fill, tip);
     }
   else
      return;

   int n = ArraySize(keep);
   ArrayResize(keep, n + 1);
   keep[n] = name;
  }

//+------------------------------------------------------------------+
void ProcessOverlayFile()
  {
   string raw = "";
   bool ok = false;
   if(InpUseCommonFiles)
      ok = FileReadAll(InpStateFile, true, raw);
   if(!ok)
      ok = FileReadAll(InpStateFile, false, raw);
   if(!ok)
      return;

   long seq = JsonExtractLong(raw, "seq");
   if(seq > 0 && seq == g_last_seq)
      return;
   g_last_seq = seq;

   string keep[];
   ArrayResize(keep, 0);

   // Iterate pattern objects arrays
   int search_from = 0;
   while(true)
     {
      int obj_key = StringFind(raw, "\"objects\"", search_from);
      if(obj_key < 0)
         break;
      int arr_open = StringFind(raw, "[", obj_key);
      if(arr_open < 0)
         break;
      int arr_close = FindMatchingBrace(raw, arr_open, '[', ']');
      if(arr_close < 0)
         break;
      string arr = StringSubstr(raw, arr_open, arr_close - arr_open + 1);
      int p = 0;
      while(true)
        {
         int o = StringFind(arr, "{", p);
         if(o < 0)
            break;
         int c = FindMatchingBrace(arr, o, '{', '}');
         if(c < 0)
            break;
         string one = StringSubstr(arr, o, c - o + 1);
         ProcessOneObject(one, keep);
         p = c + 1;
        }
      search_from = arr_close + 1;
     }

   DrawLegend(raw);
   // include legend objects already tracked via TrackDrawn during DrawLegend —
   // merge into keep
   for(int i = 0; i < ArraySize(g_drawn); i++)
     {
      if(StringFind(g_drawn[i], InpPrefix + "LEGEND") == 0)
        {
         int n = ArraySize(keep);
         ArrayResize(keep, n + 1);
         keep[n] = g_drawn[i];
        }
     }
   PruneMissing(keep);
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   // Timer-driven; tick is a light nudge for low-latency after new bar
   static datetime last_bar = 0;
   datetime t = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(t != last_bar)
     {
      last_bar = t;
      ProcessOverlayFile();
     }
  }
//+------------------------------------------------------------------+
