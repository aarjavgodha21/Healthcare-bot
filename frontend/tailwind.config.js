/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2D7FF9',
          dark: '#1E3C72',
          light: '#E8F2FF'
        },
        secondary: {
          success: '#4CAF50',
          warning: '#FFC107',
          error: '#F44336'
        },
        neutral: {
          'dark-gray': '#333333',
          'medium-gray': '#666666',
          'light-gray': '#CCCCCC',
          'off-white': '#F9FAFB'
        },
        alert: {
          'info-bg': '#2D7FF9',
          'info-text': '#FFFFFF',
          'warning-bg': '#FFF3CD',
          'warning-text': '#856404',
          'error-bg': '#F8D7DA',
          'error-text': '#721C24',
          'success-bg': '#D4EDDA',
          'success-text': '#155724'
        }
      },
      fontFamily: {
        'primary': ['Inter', 'Arial', 'Helvetica', 'sans-serif'],
        'hindi': ['Noto Sans Devanagari', 'Arial', 'Helvetica', 'sans-serif']
      },
      fontSize: {
        'h1': ['32px', '40px'],
        'h2': ['24px', '32px'],
        'h3': ['20px', '28px'],
        'body': ['16px', '24px'],
        'body-small': ['14px', '20px'],
        'caption': ['12px', '18px']
      },
      fontWeight: {
        'regular': '400',
        'medium': '500',
        'semibold': '600',
        'bold': '700'
      },
      spacing: {
        'xs': '4px',
        's': '8px',
        'm': '16px',
        'l': '24px',
        'xl': '32px',
        'xxl': '48px'
      },
      borderRadius: {
        'button': '8px',
        'card': '12px',
        'modal': '16px'
      },
      boxShadow: {
        'sm': '0 1px 3px rgba(0, 0, 0, 0.1)',
        'md': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'lg': '0 10px 25px rgba(0, 0, 0, 0.15)'
      },
      animation: {
        'fade-in': 'fadeIn 200ms ease-in',
        'slide-in-bottom': 'slideInFromBottom 300ms ease-in-out'
      },
      keyframes: {
        fadeIn: {
          'from': {
            opacity: '0',
            transform: 'translateY(10px)'
          },
          'to': {
            opacity: '1',
            transform: 'translateY(0)'
          }
        },
        slideInFromBottom: {
          'from': {
            opacity: '0',
            transform: 'translateY(100%)'
          },
          'to': {
            opacity: '1',
            transform: 'translateY(0)'
          }
        }
      },
      screens: {
        'mobile': '360px',
        'tablet': '768px',
        'desktop': '1140px'
      }
    },
  },
  plugins: [],
}
