# Diagnogenie Design System Documentation

This document defines the visual identity and UI rules for Diagnogenie — ensuring consistency across chatbot, diagnosis interface, and image upload workflows.

## 1. Brand Identity

**Product Name:** Diagnogenie

**Tone:** Trustworthy, professional, approachable, empathetic

**Design Goals:** Clean medical interface, high readability, user comfort, multilingual inclusivity (English & Hindi).

---

## 2. Color System

### 🎨 Primary Palette

**Primary Blue:** `#2D7FF9`
- **Usage:** Buttons, links, highlights, icons.

**Primary Dark:** `#1E3C72`
- **Usage:** Headers, navigation bar, strong emphasis.

**Primary Light:** `#E8F2FF`
- **Usage:** Backgrounds, hover states, subtle highlights.

### 🟢 Secondary / Accent Colors

**Success Green:** `#4CAF50`
- **Usage:** Success states, confirmation messages, healthy status.

**Warning Yellow:** `#FFC107`
- **Usage:** Alerts, pending results, image upload warnings.

**Error Red:** `#F44336`
- **Usage:** Errors, invalid input, severe alerts.

### ⚪ Neutral / Support Colors

**Dark Gray:** `#333333` (Primary text)

**Medium Gray:** `#666666` (Secondary text)

**Light Gray:** `#CCCCCC` (Borders, placeholders)

**Background White:** `#FFFFFF`

**Off White:** `#F9FAFB`

### Color Usage Guidelines

#### Do's ✅
- Use Primary Blue for all CTAs
- Use neutral colors for readability
- Keep alert colors minimal to avoid alarming users

#### Don'ts ❌
- Do not use more than two accent colors per screen
- Avoid bright red for non-error states

---

## 3. Typography

### 🔤 Font Families

**Primary Font (English):** Inter (Sans-serif, modern, highly legible)

**Secondary Font (Hindi):** Noto Sans Devanagari (ensures clear Hindi rendering)

**Fallbacks:** Arial, Helvetica, sans-serif

### 📏 Font Sizes

| Usage | Font Size | Weight | Line Height |
|-------|-----------|---------|-------------|
| H1 – Page Title | 32px | Bold | 40px |
| H2 – Section Title | 24px | Semi-bold | 32px |
| H3 – Subsection Title | 20px | Medium | 28px |
| Body Text (Primary) | 16px | Regular | 24px |
| Body Text (Secondary) | 14px | Regular | 20px |
| Caption / Meta Info | 12px | Medium | 18px |

### Typography Guidelines

#### Do's ✅
- Always left-align body text
- Maintain at least 1.5 line height for readability
- Use consistent English-Hindi pairing for bilingual displays

#### Don'ts ❌
- Don't mix more than two font weights in the same component
- Avoid using italic styles for Hindi text

---

## 4. Spacing & Layout Grid

### 📐 Grid System

- **Desktop:** 12-column grid (1140px max width)
- **Tablet:** 8-column grid (768px)
- **Mobile:** 4-column grid (360px+)

### 📏 Spacing Scale (8px baseline)

| Scale | Value | Usage |
|-------|-------|-------|
| XS | 4px | Icon padding, small gaps |
| S | 8px | Button padding, small margins |
| M | 16px | Standard spacing between components |
| L | 24px | Section spacing |
| XL | 32px | Large containers, modals |
| XXL | 48px | Major page sections |

---

## 5. Components & Guidelines

### 🔹 Buttons

**Primary Button:** Filled, `#2D7FF9` background, white text

**Secondary Button:** Outline with `#2D7FF9` border, text in primary blue

**Disabled Button:** `#CCCCCC` background, `#666666` text

#### Button Guidelines

##### Do's ✅
- Always use rounded corners: 8px radius
- Maintain 16px vertical & 24px horizontal padding

##### Don'ts ❌
- Don't overload pages with too many CTAs
- Avoid using red buttons unless for destructive actions

### 🔹 Chat Interface

**User Bubble:** Primary Blue background, white text

**Bot Bubble:** Off White background, dark gray text

**Voice Icon:** Circular, `#2D7FF9`

**Multilingual Toggle:** Always accessible at top-right

### 🔹 Image Upload Component

**Default State:** Dotted border box `#CCCCCC`, icon in `#2D7FF9`

**Hover State:** Border turns `#2D7FF9`

**Success State:** Green check `#4CAF50`

**Error State:** Red outline `#F44336`

### 🔹 Alerts

**Info:** Blue (`#2D7FF9` background, white text)

**Warning:** Yellow (`#FFF3CD` background, `#856404` text)

**Error:** Red (`#F8D7DA` background, `#721C24` text)

**Success:** Green (`#D4EDDA` background, `#155724` text)

---

## 6. Accessibility Rules

- Minimum contrast ratio: 4.5:1 for text
- Provide alt text for all images
- Ensure Hindi ↔ English toggle is visible at all times
- Voice features must have keyboard accessible equivalents

---

## 7. Iconography

- Use Lucide or Material Icons for consistency
- Stroke width: 2px
- Size variants: 16px, 24px, 32px
- Always align icons with text baseline

---

## 8. Motion & Interaction

- **Button hover:** +5% brightness
- **Chat bubble fade-in:** 200ms ease-in
- **Modal slide-in:** 300ms from bottom

---

## 9. Global Guidelines

### Do's ✅
- Keep the interface minimal and professional
- Use consistent padding and grid alignment
- Ensure multilingual support is equal in priority

### Don'ts ❌
- Don't overload with unnecessary animations
- Avoid inconsistent colors between different modules
- Never use jargon-heavy text for medical advice — keep it clear and empathetic

---

## 10. Implementation Reference

### CSS Variables
```css
:root {
  /* Primary Colors */
  --primary-blue: #2D7FF9;
  --primary-dark: #1E3C72;
  --primary-light: #E8F2FF;
  
  /* Secondary Colors */
  --success-green: #4CAF50;
  --warning-yellow: #FFC107;
  --error-red: #F44336;
  
  /* Neutral Colors */
  --dark-gray: #333333;
  --medium-gray: #666666;
  --light-gray: #CCCCCC;
  --background-white: #FFFFFF;
  --off-white: #F9FAFB;
  
  /* Typography */
  --font-primary: 'Inter', Arial, Helvetica, sans-serif;
  --font-hindi: 'Noto Sans Devanagari', Arial, Helvetica, sans-serif;
  
  /* Spacing */
  --space-xs: 4px;
  --space-s: 8px;
  --space-m: 16px;
  --space-l: 24px;
  --space-xl: 32px;
  --space-xxl: 48px;
  
  /* Border Radius */
  --radius-button: 8px;
  --radius-card: 12px;
  --radius-modal: 16px;
}
```

### Tailwind Config Integration
For your existing Tailwind setup, extend the configuration with these design tokens:

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2D7FF9',
          dark: '#1E3C72',
          light: '#E8F2FF'
        },
        success: '#4CAF50',
        warning: '#FFC107',
        error: '#F44336',
        gray: {
          dark: '#333333',
          medium: '#666666',
          light: '#CCCCCC'
        }
      },
      fontFamily: {
        'primary': ['Inter', 'Arial', 'Helvetica', 'sans-serif'],
        'hindi': ['Noto Sans Devanagari', 'Arial', 'Helvetica', 'sans-serif']
      },
      spacing: {
        'xs': '4px',
        's': '8px',
        'm': '16px',
        'l': '24px',
        'xl': '32px',
        'xxl': '48px'
      }
    }
  }
}
```

---

*This design system should be referenced for all UI development across the Diagnogenie platform to ensure consistency and optimal user experience.*