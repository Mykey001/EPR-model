//+------------------------------------------------------------------+
//|          Combined HL-EPR Mean Reversion Sniper EA                |
//|  v2.0 — Swing Point Labels + Swing SL + Risk:Reward TP          |
//|  Logic: Pivot High/Low + EMA crossover + PSAR + RSI + BB + ADX  |
//+------------------------------------------------------------------+
#property copyright "Converted from PineScript"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//=== INPUT GROUPS ================================================================

// --- Pivot Points ---
input group "=== Pivot Points High / Low ==="
input int    InpLeftLenH   = 10;    // Pivot High - Left Bars
input int    InpRightLenH  = 10;    // Pivot High - Right Bars
input int    InpLeftLenL   = 10;    // Pivot Low  - Left Bars
input int    InpRightLenL  = 10;    // Pivot Low  - Right Bars

// --- Swing Point Labels ---
input group "=== Swing Point Labels ==="
input bool   InpShowSwingLabels  = true;       // Show Swing Point Arrows & Labels
input color  InpPivotHighColor   = clrRed;     // Pivot High Arrow Color
input color  InpPivotLowColor    = clrLime;    // Pivot Low  Arrow Color
input int    InpLabelFontSize    = 8;          // Label Font Size
input int    InpArrowSize        = 2;          // Arrow Size (1-5)
input int    InpMaxSwingLabels   = 100;        // Max Labels to Keep on Chart

// --- EMA Settings ---
input group "=== EMA Settings ==="
input int    InpEmaFastLen = 5;     // Fast EMA Length
input int    InpEmaSlowLen = 15;    // Slow EMA Length

// --- PSAR Settings ---
input group "=== PSAR Settings ==="
input double InpSarStart   = 0.02;  // PSAR Start
input double InpSarInc     = 0.02;  // PSAR Increment
input double InpSarMax     = 0.20;  // PSAR Max Value

// --- RSI Settings ---
input group "=== RSI Settings (Mean Reversion) ==="
input int    InpRsiLen     = 7;     // RSI Length
input int    InpRsiOB      = 70;    // RSI Overbought
input int    InpRsiOS      = 30;    // RSI Oversold

// --- Macro Filter ---
input group "=== Macro Filter ==="
input ENUM_TIMEFRAMES InpMacroTf     = PERIOD_H1;  // Macro Timeframe
input int             InpMacroEmaLen = 200;          // Macro EMA Length
input bool            InpUseMacro    = true;          // Enable Macro Filter

// --- Volatility / ADX Filter ---
input group "=== Volatility Filter ==="
input int    InpBBLen        = 20;    // Bollinger Bands Length
input double InpBBMult       = 2.0;   // Bollinger Bands StdDev Multiplier
input int    InpAdxThreshold = 30;    // ADX Max Threshold (mean reversion needs low ADX)
input bool   InpUseAdx       = true;  // Enable ADX Filter

// --- Trade Management ---
input group "=== Trade Management ==="
input double InpLotSize          = 0.1;    // Lot Size
input bool   InpUseSwingPointSL  = true;   // Use Swing Point Stop Loss (else fixed pips)
input double InpSwingSlBufferPips = 5.0;   // Swing SL Buffer – extra pips beyond swing (safety margin)
input double InpRiskRewardRatio  = 2.0;    // Risk:Reward Ratio for Take Profit (e.g. 2 = 1:2)
input double InpStopLossPips     = 50.0;   // Fixed Stop Loss (pips) – used when swing SL is OFF
input int    InpMagicNumber      = 202406; // Magic Number
input int    InpSlippage         = 10;     // Max Slippage (points)

// --- Lookback for "was extreme" ---
input group "=== Mean Reversion Lookback ==="
input int    InpExtLookback  = 5;     // Bars since extreme condition (Pine: barssince <= 5)

//=== GLOBAL HANDLES ==============================================================

int g_hEmaFast   = INVALID_HANDLE;
int g_hEmaSlow   = INVALID_HANDLE;
int g_hSar       = INVALID_HANDLE;
int g_hRsi       = INVALID_HANDLE;
int g_hBB        = INVALID_HANDLE;
int g_hAdx       = INVALID_HANDLE;
int g_hMacroEma  = INVALID_HANDLE;

CTrade g_trade;

//=== SWING POINT STATE ===========================================================

double   g_lastPivotHigh      = 0.0;   // Most recent confirmed pivot high price
double   g_lastPivotLow       = 0.0;   // Most recent confirmed pivot low price
datetime g_lastPivotHighTime  = 0;     // Bar time of last pivot high
datetime g_lastPivotLowTime   = 0;     // Bar time of last pivot low

// Track how many labels we've drawn so we can clean up old ones
int      g_labelCount         = 0;

//=== INIT ========================================================================

int OnInit()
{
   // Fast EMA
   g_hEmaFast = iMA(_Symbol, PERIOD_CURRENT, InpEmaFastLen, 0, MODE_EMA, PRICE_CLOSE);
   if(g_hEmaFast == INVALID_HANDLE) { Print("ERROR: Fast EMA handle"); return INIT_FAILED; }

   // Slow EMA
   g_hEmaSlow = iMA(_Symbol, PERIOD_CURRENT, InpEmaSlowLen, 0, MODE_EMA, PRICE_CLOSE);
   if(g_hEmaSlow == INVALID_HANDLE) { Print("ERROR: Slow EMA handle"); return INIT_FAILED; }

   // Parabolic SAR
   g_hSar = iSAR(_Symbol, PERIOD_CURRENT, InpSarStart, InpSarMax);
   if(g_hSar == INVALID_HANDLE) { Print("ERROR: SAR handle"); return INIT_FAILED; }

   // RSI
   g_hRsi = iRSI(_Symbol, PERIOD_CURRENT, InpRsiLen, PRICE_CLOSE);
   if(g_hRsi == INVALID_HANDLE) { Print("ERROR: RSI handle"); return INIT_FAILED; }

   // Bollinger Bands
   g_hBB = iBands(_Symbol, PERIOD_CURRENT, InpBBLen, 0, InpBBMult, PRICE_CLOSE);
   if(g_hBB == INVALID_HANDLE) { Print("ERROR: BB handle"); return INIT_FAILED; }

   // ADX — buffer 0=ADX, 1=+DI, 2=-DI
   g_hAdx = iADX(_Symbol, PERIOD_CURRENT, 14);
   if(g_hAdx == INVALID_HANDLE) { Print("ERROR: ADX handle"); return INIT_FAILED; }

   // Macro EMA
   g_hMacroEma = iMA(_Symbol, InpMacroTf, InpMacroEmaLen, 0, MODE_EMA, PRICE_CLOSE);
   if(g_hMacroEma == INVALID_HANDLE) { Print("ERROR: Macro EMA handle"); return INIT_FAILED; }

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(InpSlippage);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   // Remove any leftover labels from a previous run
   RemoveAllSwingLabels();

   Print("Combined HL-EPR EA v2.0 initialized. SwingSL=", InpUseSwingPointSL,
         " RR=1:", InpRiskRewardRatio);
   return INIT_SUCCEEDED;
}

//=== DEINIT ======================================================================

void OnDeinit(const int reason)
{
   IndicatorRelease(g_hEmaFast);
   IndicatorRelease(g_hEmaSlow);
   IndicatorRelease(g_hSar);
   IndicatorRelease(g_hRsi);
   IndicatorRelease(g_hBB);
   IndicatorRelease(g_hAdx);
   IndicatorRelease(g_hMacroEma);

   // Clean up chart objects on removal (keep on chart-reattach for reference)
   if(reason == REASON_REMOVE)
      RemoveAllSwingLabels();
}

//=== UTILITY: safe buffer copy ===================================================

bool GetBuffer(int handle, int bufferIdx, int startPos, int count, double &arr[])
{
   ArraySetAsSeries(arr, true);
   int copied = CopyBuffer(handle, bufferIdx, startPos, count, arr);
   return (copied == count);
}

//=== SWING LABEL HELPERS =========================================================

// Prefix for all objects created by this EA so we can selectively clean them up
string ObjPrefix() { return "SWNG_" + IntegerToString(InpMagicNumber) + "_"; }

void RemoveAllSwingLabels()
{
   string prefix = ObjPrefix();
   int total = ObjectsTotal(0, -1, -1);
   // Iterate backwards so deletion doesn't shift indices
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(0, i, -1, -1);
      if(StringFind(name, prefix) == 0)
         ObjectDelete(0, name);
   }
   g_labelCount = 0;
}

//--- Draw a pivot HIGH label: down-arrow above the bar + price text
void DrawPivotHighLabel(datetime barTime, double price)
{
   if(!InpShowSwingLabels) return;

   // Build unique object names (arrow + text)
   string arrowName = ObjPrefix() + "PH_ARR_" + IntegerToString((int)barTime);
   string textName  = ObjPrefix() + "PH_TXT_" + IntegerToString((int)barTime);

   // Arrow: down-pointing arrow placed slightly above the high
   double arrowPrice = price + 15 * _Point;   // small offset above bar tip

   if(!ObjectCreate(0, arrowName, OBJ_ARROW, 0, barTime, arrowPrice))
   {
      // Object may already exist (re-attach scenario) — just move it
      ObjectMove(0, arrowName, 0, barTime, arrowPrice);
   }
   ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, 234);  // WINGDINGS down arrow
   ObjectSetInteger(0, arrowName, OBJPROP_COLOR,     InpPivotHighColor);
   ObjectSetInteger(0, arrowName, OBJPROP_WIDTH,     InpArrowSize);
   ObjectSetInteger(0, arrowName, OBJPROP_ANCHOR,    ANCHOR_BOTTOM);
   ObjectSetInteger(0, arrowName, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, arrowName, OBJPROP_HIDDEN,    true);

   // Price text label
   string priceStr = "PH " + DoubleToString(price, _Digits);
   if(!ObjectCreate(0, textName, OBJ_TEXT, 0, barTime, arrowPrice + 8 * _Point))
      ObjectMove(0, textName, 0, barTime, arrowPrice + 8 * _Point);
   ObjectSetString (0, textName, OBJPROP_TEXT,      priceStr);
   ObjectSetInteger(0, textName, OBJPROP_COLOR,     InpPivotHighColor);
   ObjectSetInteger(0, textName, OBJPROP_FONTSIZE,  InpLabelFontSize);
   ObjectSetInteger(0, textName, OBJPROP_ANCHOR,    ANCHOR_LEFT_LOWER);
   ObjectSetInteger(0, textName, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, textName, OBJPROP_HIDDEN,    true);

   g_labelCount += 2;
   ChartRedraw(0);

   // Housekeeping: prune oldest labels if we exceed the cap
   if(g_labelCount > InpMaxSwingLabels * 2)
      PruneOldestSwingLabel();
}

//--- Draw a pivot LOW label: up-pointing arrow below the bar + price text
void DrawPivotLowLabel(datetime barTime, double price)
{
   if(!InpShowSwingLabels) return;

   string arrowName = ObjPrefix() + "PL_ARR_" + IntegerToString((int)barTime);
   string textName  = ObjPrefix() + "PL_TXT_" + IntegerToString((int)barTime);

   double arrowPrice = price - 15 * _Point;  // small offset below bar

   if(!ObjectCreate(0, arrowName, OBJ_ARROW, 0, barTime, arrowPrice))
      ObjectMove(0, arrowName, 0, barTime, arrowPrice);
   ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, 233);  // WINGDINGS up arrow
   ObjectSetInteger(0, arrowName, OBJPROP_COLOR,     InpPivotLowColor);
   ObjectSetInteger(0, arrowName, OBJPROP_WIDTH,     InpArrowSize);
   ObjectSetInteger(0, arrowName, OBJPROP_ANCHOR,    ANCHOR_TOP);
   ObjectSetInteger(0, arrowName, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, arrowName, OBJPROP_HIDDEN,    true);

   string priceStr = "PL " + DoubleToString(price, _Digits);
   if(!ObjectCreate(0, textName, OBJ_TEXT, 0, barTime, arrowPrice - 8 * _Point))
      ObjectMove(0, textName, 0, barTime, arrowPrice - 8 * _Point);
   ObjectSetString (0, textName, OBJPROP_TEXT,      priceStr);
   ObjectSetInteger(0, textName, OBJPROP_COLOR,     InpPivotLowColor);
   ObjectSetInteger(0, textName, OBJPROP_FONTSIZE,  InpLabelFontSize);
   ObjectSetInteger(0, textName, OBJPROP_ANCHOR,    ANCHOR_LEFT_UPPER);
   ObjectSetInteger(0, textName, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, textName, OBJPROP_HIDDEN,    true);

   g_labelCount += 2;
   ChartRedraw(0);

   if(g_labelCount > InpMaxSwingLabels * 2)
      PruneOldestSwingLabel();
}

//--- Remove the single oldest swing label pair to keep chart tidy
void PruneOldestSwingLabel()
{
   string prefix = ObjPrefix();
   datetime oldest = D'3000.01.01';
   string  oldestName = "";

   int total = ObjectsTotal(0, -1, -1);
   for(int i = 0; i < total; i++)
   {
      string name = ObjectName(0, i, -1, -1);
      if(StringFind(name, prefix) != 0) continue;

      datetime t = (datetime)ObjectGetInteger(0, name, OBJPROP_TIME);
      if(t < oldest)
      {
         oldest     = t;
         oldestName = name;
      }
   }
   if(oldestName != "")
   {
      ObjectDelete(0, oldestName);
      g_labelCount--;
   }
}

//=== PIVOT HIGH/LOW DETECTION ====================================================

bool IsPivotHigh(const double &highs[], int offset, int leftLen, int rightLen)
{
   double candidate = highs[offset];
   for(int i = 1; i <= leftLen; i++)
      if(highs[offset + i] >= candidate) return false;
   for(int i = 1; i <= rightLen; i++)
      if(highs[offset - i] >= candidate) return false;
   return true;
}

bool IsPivotLow(const double &lows[], int offset, int leftLen, int rightLen)
{
   double candidate = lows[offset];
   for(int i = 1; i <= leftLen; i++)
      if(lows[offset + i] <= candidate) return false;
   for(int i = 1; i <= rightLen; i++)
      if(lows[offset - i] <= candidate) return false;
   return true;
}

//=== POSITION MANAGEMENT =========================================================

bool HasOpenPosition(ENUM_POSITION_TYPE posType)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC)  == InpMagicNumber &&
            PositionGetInteger(POSITION_TYPE)   == posType)
            return true;
   }
   return false;
}

//=== STOP LOSS & TAKE PROFIT CALCULATION =========================================

// Returns the swing-point SL for a BUY order.
// SL = last pivot low  minus buffer pips.
// If no pivot low recorded yet, falls back to fixed pips.
double CalcBuySL(double entryPrice)
{
   double pipSize = _Point * 10;

   if(InpUseSwingPointSL && g_lastPivotLow > 0.0)
   {
      double sl = g_lastPivotLow - InpSwingSlBufferPips * pipSize;
      sl = NormalizeDouble(sl, _Digits);

      // Safety: ensure SL is actually below entry
      if(sl < entryPrice)
         return sl;

      // Pivot low is above entry (stale data) — fall through to fixed
      Print("WARNING: Swing Low SL (", sl, ") >= entry (", entryPrice,
            "). Falling back to fixed pip SL.");
   }

   // Fixed pip fallback
   return NormalizeDouble(entryPrice - InpStopLossPips * pipSize, _Digits);
}

// Returns the swing-point SL for a SELL order.
// SL = last pivot high plus buffer pips.
double CalcSellSL(double entryPrice)
{
   double pipSize = _Point * 10;

   if(InpUseSwingPointSL && g_lastPivotHigh > 0.0)
   {
      double sl = g_lastPivotHigh + InpSwingSlBufferPips * pipSize;
      sl = NormalizeDouble(sl, _Digits);

      if(sl > entryPrice)
         return sl;

      Print("WARNING: Swing High SL (", sl, ") <= entry (", entryPrice,
            "). Falling back to fixed pip SL.");
   }

   return NormalizeDouble(entryPrice + InpStopLossPips * pipSize, _Digits);
}

// Calculate TP from entry and SL using the selected Risk:Reward ratio.
double CalcTP(double entryPrice, double slPrice)
{
   double slDistance = MathAbs(entryPrice - slPrice);
   double tpDistance = slDistance * InpRiskRewardRatio;

   if(entryPrice > slPrice)                        // BUY
      return NormalizeDouble(entryPrice + tpDistance, _Digits);
   else                                             // SELL
      return NormalizeDouble(entryPrice - tpDistance, _Digits);
}

//=== MAIN TICK ===================================================================

void OnTick()
{
   // Only act on new bar to match Pine bar-close logic
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBar == lastBar) return;
   lastBar = currentBar;

   //--- How many bars we need ---
   int maxLen    = MathMax(InpLeftLenH + InpRightLenH, InpLeftLenL + InpRightLenL) + 10;
   int barsNeeded = MathMax(maxLen, InpExtLookback + InpBBLen + 5);
   barsNeeded = MathMax(barsNeeded, InpMacroEmaLen + 5);

   //--- Load price arrays ---
   double highArr[], lowArr[], closeArr[];
   ArraySetAsSeries(highArr,  true);
   ArraySetAsSeries(lowArr,   true);
   ArraySetAsSeries(closeArr, true);

   int copiedH = CopyHigh (_Symbol, PERIOD_CURRENT, 0, barsNeeded, highArr);
   int copiedL = CopyLow  (_Symbol, PERIOD_CURRENT, 0, barsNeeded, lowArr);
   int copiedC = CopyClose(_Symbol, PERIOD_CURRENT, 0, barsNeeded, closeArr);
   if(copiedH < barsNeeded || copiedL < barsNeeded || copiedC < barsNeeded)
   { Print("Not enough bars yet."); return; }

   //--- Swing Point Detection (check the confirmed pivot at bar[rightLen]) ---
   // The confirmed pivot sits at bar index = rightLen (it has rightLen bars to its right)
   int pivotIdxH = InpRightLenH;
   int pivotIdxL = InpRightLenL;

   bool pivotHigh = IsPivotHigh(highArr, pivotIdxH, InpLeftLenH, InpRightLenH);
   bool pivotLow  = IsPivotLow (lowArr,  pivotIdxL, InpLeftLenL, InpRightLenL);

   // Retrieve bar time for labelling
   if(pivotHigh)
   {
      datetime pivotHighTime = iTime(_Symbol, PERIOD_CURRENT, pivotIdxH);
      double   pivotHighPrice = highArr[pivotIdxH];

      // Only update & draw if this is a NEW pivot high we haven't logged yet
      if(pivotHighTime != g_lastPivotHighTime)
      {
         g_lastPivotHigh     = pivotHighPrice;
         g_lastPivotHighTime = pivotHighTime;

         DrawPivotHighLabel(pivotHighTime, pivotHighPrice);
         Print("✦ Pivot HIGH | Price=", pivotHighPrice,
               " | Time=", TimeToString(pivotHighTime));
      }
   }

   if(pivotLow)
   {
      datetime pivotLowTime  = iTime(_Symbol, PERIOD_CURRENT, pivotIdxL);
      double   pivotLowPrice = lowArr[pivotIdxL];

      if(pivotLowTime != g_lastPivotLowTime)
      {
         g_lastPivotLow     = pivotLowPrice;
         g_lastPivotLowTime = pivotLowTime;

         DrawPivotLowLabel(pivotLowTime, pivotLowPrice);
         Print("✦ Pivot LOW  | Price=", pivotLowPrice,
               " | Time=", TimeToString(pivotLowTime));
      }
   }

   //--- EMA Fast & Slow ---
   double emaFastArr[], emaSlowArr[];
   if(!GetBuffer(g_hEmaFast, 0, 0, 3, emaFastArr)) return;
   if(!GetBuffer(g_hEmaSlow, 0, 0, 3, emaSlowArr)) return;

   double emaFast1 = emaFastArr[1], emaFast2 = emaFastArr[2];
   double emaSlow1 = emaSlowArr[1], emaSlow2 = emaSlowArr[2];

   bool crossOver  = (emaFast2 <= emaSlow2) && (emaFast1 > emaSlow1);
   bool crossUnder = (emaFast2 >= emaSlow2) && (emaFast1 < emaSlow1);

   //--- PSAR ---
   double sarArr[];
   if(!GetBuffer(g_hSar, 0, 0, 3, sarArr)) return;
   double sar1    = sarArr[1];
   bool psarBull  = (sar1 < closeArr[1]);
   bool psarBear  = (sar1 > closeArr[1]);

   //--- RSI ---
   double rsiArr[];
   if(!GetBuffer(g_hRsi, 0, 0, InpExtLookback + 3, rsiArr)) return;

   //--- Bollinger Bands ---
   double bbUpperArr[], bbLowerArr[];
   if(!GetBuffer(g_hBB, 1, 0, InpExtLookback + 3, bbUpperArr)) return;
   if(!GetBuffer(g_hBB, 2, 0, InpExtLookback + 3, bbLowerArr)) return;

   bool wasExtOS = false, wasExtOB = false;
   for(int k = 1; k <= InpExtLookback; k++)
   {
      if((lowArr[k]  <= bbLowerArr[k]) || (rsiArr[k] <= InpRsiOS)) wasExtOS = true;
      if((highArr[k] >= bbUpperArr[k]) || (rsiArr[k] >= InpRsiOB)) wasExtOB = true;
   }

   //--- ADX Filter ---
   double adxArr[];
   if(!GetBuffer(g_hAdx, 0, 0, 3, adxArr)) return;
   double adxVal = adxArr[1];
   bool adxOk    = !InpUseAdx || (adxVal < InpAdxThreshold);

   //--- Macro Filter ---
   double macroEmaArr[], macroPriceArr[];
   ArraySetAsSeries(macroEmaArr,   true);
   ArraySetAsSeries(macroPriceArr, true);
   if(CopyBuffer(g_hMacroEma, 0, 0, 3, macroEmaArr)   < 3) return;
   if(CopyClose(_Symbol, InpMacroTf, 0, 3, macroPriceArr) < 3) return;

   double macroEma1   = macroEmaArr[1];
   double macroPrice1 = macroPriceArr[0];

   bool macroOkBull = !InpUseMacro || (macroPrice1 > macroEma1);
   bool macroOkBear = !InpUseMacro || (macroPrice1 < macroEma1);

   //--- Final Signal Logic ---
   bool buySig  = crossOver  && psarBull && wasExtOS && adxOk && macroOkBull;
   bool sellSig = crossUnder && psarBear && wasExtOB && adxOk && macroOkBear;

   //--- Trade Execution ---
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(buySig && !HasOpenPosition(POSITION_TYPE_BUY))
   {
      double sl = CalcBuySL(ask);
      double tp = CalcTP(ask, sl);
      double slPips = MathAbs(ask - sl) / (_Point * 10);
      double tpPips = MathAbs(tp  - ask) / (_Point * 10);

      Print("BUY Signal | SwingSL=", InpUseSwingPointSL,
            " | LastPivotLow=",  g_lastPivotLow,
            " | SL=", sl, " (", DoubleToString(slPips, 1), " pips)",
            " | TP=", tp, " (", DoubleToString(tpPips, 1), " pips)",
            " | RR=1:", InpRiskRewardRatio);

      if(g_trade.Buy(InpLotSize, _Symbol, ask, sl, tp, "EPR Buy | RR1:" +
                     DoubleToString(InpRiskRewardRatio, 1)))
         Print("BUY opened. ADX=", adxVal, " macroPrice=", macroPrice1, " macroEMA=", macroEma1);
      else
         Print("BUY failed: ", g_trade.ResultRetcodeDescription());
   }

   if(sellSig && !HasOpenPosition(POSITION_TYPE_SELL))
   {
      double sl = CalcSellSL(bid);
      double tp = CalcTP(bid, sl);
      double slPips = MathAbs(sl  - bid) / (_Point * 10);
      double tpPips = MathAbs(bid - tp)  / (_Point * 10);

      Print("SELL Signal | SwingSL=", InpUseSwingPointSL,
            " | LastPivotHigh=", g_lastPivotHigh,
            " | SL=", sl, " (", DoubleToString(slPips, 1), " pips)",
            " | TP=", tp, " (", DoubleToString(tpPips, 1), " pips)",
            " | RR=1:", InpRiskRewardRatio);

      if(g_trade.Sell(InpLotSize, _Symbol, bid, sl, tp, "EPR Sell | RR1:" +
                      DoubleToString(InpRiskRewardRatio, 1)))
         Print("SELL opened. ADX=", adxVal, " macroPrice=", macroPrice1, " macroEMA=", macroEma1);
      else
         Print("SELL failed: ", g_trade.ResultRetcodeDescription());
   }
}

//=== END OF FILE =================================================================
